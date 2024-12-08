#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
import shutil
import time

from utils.compression import create_octree, decode_oct ,create_octree_train, d1halfing_fast
from utils.quant_utils import VanillaQuan, Quant_all, split_length, seg_quant, seg_quant_forward, seg_quant_reverse, dp_split
from raht_torch import copyAsort, haar3D_param, inv_haar3D_param, itransform_batched_torch, transform_batched_torch
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 raht: bool = False,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        
        # add by mesongs
        self.raht = raht
        self.feature_channels = -1

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()


    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_scaling_v(self):
        return 1.0*self.scaling_activation(self._scaling_v)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_rotation_v(self):
        return self.rotation_activation(self._rotation_v)
    
    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_anchor_v(self):
        return self._anchor_v
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_covariance_v(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling_v, scaling_modifier, self._rotation_v)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    def set_spatial_lr_scale(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        
        print('create from pcd')
        print('size of different attributes')
        print('_anchor', self._anchor.shape)
        print('anchor_feat', self._anchor_feat.shape)
        print('_offset', self._offset.shape)
        print('_opacity', self._opacity.shape)
        print('_scaling', self._scaling.shape)
        print('_rotation', self._rotation.shape)
        


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        print('self.spatial_lr_scale!!!!!!!!!!!',self.spatial_lr_scale)
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps,
                                                    start_steps=training_args.position_lr_start_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr

            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def apply_raht_forward(self):
        rf = torch.concat([
            self._anchor_feat_v.detach(), 
            self._offset_v.detach().flatten(-2).contiguous(), 
            self._opacity_v.detach().contiguous(), 
            self._scaling_v.detach(), 
            self._rotation_v.detach()], -1)

        C = rf[self.reorder]
        iW1 = self.res['iW1']
        iW2 = self.res['iW2']
        iLeft_idx = self.res['iLeft_idx']
        iRight_idx = self.res['iRight_idx']
        
        for d in range(self.depth * 3):
            w1 = iW1[d]
            w2 = iW2[d]
            left_idx = iLeft_idx[d]
            right_idx = iRight_idx[d]
            C[left_idx], C[right_idx] = transform_batched_torch(w1, 
                                                  w2, 
                                                  C[left_idx], 
                                                  C[right_idx])
        
        return C
    
    def apply_raht_reverse(self, V, C, depth):
        w, val, reorder = copyAsort(V)        
        res_inv = inv_haar3D_param(V, depth)
        pos = res_inv['pos']
        iW1 = res_inv['iW1']
        iW2 = res_inv['iW2']
        iS = res_inv['iS']
        
        iLeft_idx = res_inv['iLeft_idx']
        iRight_idx = res_inv['iRight_idx']
    
        iLeft_idx_CT = res_inv['iLeft_idx_CT']
        iRight_idx_CT = res_inv['iRight_idx_CT']
        iTrans_idx = res_inv['iTrans_idx']
        iTrans_idx_CT = res_inv['iTrans_idx_CT']

        CT_yuv_q_temp = C[pos.astype(int)]
        draht_features = torch.zeros(C.shape, dtype=torch.float32).cuda()
        OC = torch.zeros(C.shape, dtype=torch.float32).cuda()
        
        for i in range(depth*3):
            w1 = iW1[i]
            w2 = iW2[i]
            S = iS[i]
            
            left_idx, right_idx = iLeft_idx[i], iRight_idx[i]
            left_idx_CT, right_idx_CT = iLeft_idx_CT[i], iRight_idx_CT[i]
            
            trans_idx, trans_idx_CT = iTrans_idx[i], iTrans_idx_CT[i]
            
            
            OC[trans_idx] = CT_yuv_q_temp[trans_idx_CT]
            OC[left_idx], OC[right_idx] = itransform_batched_torch(w1, 
                                                    w2, 
                                                    CT_yuv_q_temp[left_idx_CT], 
                                                    CT_yuv_q_temp[right_idx_CT])  
            CT_yuv_q_temp[:S] = OC[:S]

        draht_features[reorder] = OC
        
        return draht_features
        
    
    def apply_quant_forward(self, f):
        qa_cnt = 0
        ret = torch.zeros_like(f)
        trans = []
        # splits = split_length(f.shape[0], self.n_block)
        splits = self.dp_split
        assert f.shape[0]==sum(splits[0]), "split length is error"
        for i in range(f.shape[-1]):
            qfi, sz = seg_quant_forward(f[:, i], splits[i], self.qas[qa_cnt : qa_cnt + self.n_block])
            qa_cnt += self.n_block
            ret[:, i] = qfi.reshape(-1).cpu()
            trans.extend(sz)
            
        return ret, trans
    
    def apply_quant_reverse(self, f, trans_array, p, n_block):
        ret = []
        # split = split_length(f.shape[0], n_block)
        split = self.dp_split
        assert f.shape[0]==sum(split[0]), "split length is error"
        for i in range(f.shape[-1]):
            ri = seg_quant_reverse(f[:, i], split[i], trans_array[p:p+2*n_block])
            ret.append(ri.reshape(-1, 1))
            p += 2*n_block 
        ret = torch.concat(ret, dim=-1)
        return ret
        
    def save_compressed(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        bin_dir = os.path.join(save_dir, 'bins')
        os.makedirs(bin_dir, exist_ok=True)
        
        trans = []
        trans.extend([self.depth, self.n_block])
        
        features = self.apply_raht_forward()
        # features = torch.concat([
        #     self._anchor_feat_v.detach(), 
        #     self._offset_v.detach().flatten(-2).contiguous(), 
        #     self._opacity_v.detach().contiguous(), 
        #     self._scaling_v.detach(), 
        #     self._rotation_v.detach()], -1)

        # ff = features
        ff = features[0]
        qf, sz = self.apply_quant_forward(features[1:])
        trans.extend(sz)
        qf_16 = qf[:,self.high_bit_channels]
        qf_8 = qf[:,self.low_bit_channels]
        np_split_0 = np.array(self.dp_split[0])
        np_split_1 = np.array(self.dp_split[32])
        
        # print("save")
        # print(self.oct.shape
        np.savez_compressed(os.path.join(bin_dir, 'oct'), points=self.oct, params=self.oct_param)
        # np.savez_compressed(os.path.join(bin_dir, 'oct'), points=self.oct, params=self.oct_param, xyz = self._anchor_v.cpu().numpy())
        # np.savez_compressed(os.path.join(bin_dir, 'fe'), f=ff.cpu().numpy(), i_16=qf_16.cpu().numpy().astype(np.uint16), t=trans)#这里需要改一下
        # np.savez_compressed(os.path.join(bin_dir, 'fe'), f=ff.cpu().numpy(), t=trans)#这里需要改一下
        np.savez_compressed(os.path.join(bin_dir, 'fe'), f=ff.cpu().numpy(), i_8=qf_8.cpu().numpy().astype(np.uint8), i_16=qf_16.cpu().numpy().astype(np.uint16), t=trans, h=self.high_bit_channels, s0=np_split_0, s1=np_split_1)#这里需要改一下
        self.save_mlp_checkpoints(bin_dir)
        
        bin_zip_path = os.path.join(save_dir, 'bins.zip')
        os.system(f'zip -j {bin_zip_path} {bin_dir}/*')
        zip_file_size = os.path.getsize(bin_zip_path)
        
        print('final sum:', zip_file_size , 'B')
        print('final sum:', zip_file_size / 1024, 'KB')
        print('final sum:', zip_file_size / 1024 / 1024, 'MB')

        return zip_file_size / 1024 / 1024

        
    def load_compressed(self, save_path):
        t = time.time()
        if save_path[-4:] == '.zip':
            print('Assume the input file path is a .zip file')
            
            bin_dir = os.path.join('/'.join(save_path.split('/')[:-1]), 'bins_tmp')
            if os.path.exists(bin_dir):
                shutil.rmtree(bin_dir)
            os.system(f'unzip {save_path} -d {bin_dir}')
        else:
            print('Assume the input file path is a dir that contains a \'bins\' dir')
            bin_dir = os.path.join(save_path, 'bins')
        print('load ply from:', bin_dir)
        
        oct_vals = np.load(os.path.join(bin_dir, 'oct.npz'))
        octree = oct_vals["points"]
        oct_param = oct_vals["params"]
        # xyz = oct_vals["xyz"]
        
        attris = np.load(os.path.join(bin_dir, 'fe.npz'))
        high = attris['h']
        low = [i for i in range(73) if i not in high]
        ff = torch.tensor(attris['f'], dtype=torch.float32).cuda()

        i_8 = torch.tensor(attris['i_8'], dtype=torch.float32).cuda()

        i_16_np = attris['i_16'].astype(np.int32)
        i_16 = torch.tensor(i_16_np, dtype=torch.float32).cuda()

        qf = torch.zeros((i_8.shape[0], len(high) + len(low))).cuda()
        qf[:, low] = i_8
        qf[:, high] = i_16
        trans = attris['t']
        split_0 = attris['s0'].tolist()
        split_1 = attris['s1'].tolist()
        self.dp_split = [split_0]*32 + [split_1]*41
        
        depth = int(trans[0])
        n_block = int(trans[1])
        self.depth = depth
        self.n_block = n_block
        
        dxyz, V = decode_oct(oct_param, octree, depth)
        # dxyz = xyz

        dqf = self.apply_quant_reverse(qf, trans, 2, n_block)
        fe = torch.concat([ff.reshape(1, -1), dqf], dim=0)
        
        features = self.apply_raht_reverse(V, fe, depth)
        # features = ff
        # dxyz = xyz
        
        self._anchor = nn.Parameter(torch.tensor(dxyz, dtype=torch.float, device="cuda").requires_grad_(False))
        base = 0
        self._anchor_feat = nn.Parameter(features[:, base:base + self.feat_dim].cuda().contiguous().requires_grad_(True))
        base += self.feat_dim #47
        self._offset = nn.Parameter(features[:, base:base + self.n_offsets*3].reshape([-1, self.n_offsets, 3]).cuda().contiguous().requires_grad_(True))
        base += self.n_offsets*3  #62
        self._opacity = nn.Parameter(features[:, base:base+1].cuda().requires_grad_(True))
        base += 1 #63
        self._scaling = nn.Parameter(features[:, base:base+6].cuda().requires_grad_(True))
        base += 6 #69
        self._rotation = nn.Parameter(features[:, base:base+4].cuda().requires_grad_(True))
        base += 4 #73
        print('base, feature size', base)
        self.feature_channels = base
        
        self.load_mlp_checkpoints(bin_dir)
        
        torch.cuda.empty_cache()
        print('load time', time.time()-t)
        

    def init_qas(self, n_block, qbits):
        assert self.feature_channels != -1
        self.n_block = n_block
        # n_qs = self.feature_channels * n_block
        self.qas = nn.ModuleList([])
        # for i in range(n_qs): 
        #     self.qas.append(VanillaQuan(bit=8).cuda())
        high_bit_channels = []
        low_bit_channels = []
        for i in range(self.feature_channels):
            qbit = qbits[i]
            if qbit > 8:
                high_bit_channels.append(i)
            else:
                low_bit_channels.append(i)
            for j in range(n_block):
                self.qas.append(VanillaQuan(bit=qbit).cuda())
        self.high_bit_channels = high_bit_channels
        self.low_bit_channels = low_bit_channels
        # print('Init qa, length:', n_qs)
    
    
    
    def octree_coding(self, imp, merge_type, raht=False):
        
        print('begin octree coding')
        print('size of different attributes')
        print('_anchor', self._anchor.shape)
        print('anchor_feat', self._anchor_feat.shape)
        print('_offset', self._offset.shape)
        print('_opacity', self._opacity.shape)
        print('_scaling', self._scaling.shape)
        print('_rotation', self._rotation.shape)
        
        # np.save('xyz.npy', self._anchor.detach().cpu().numpy())

        features = torch.concat([
            self._anchor_feat.detach(), 
            self._offset.detach().flatten(-2).contiguous(), 
            self._opacity.detach().contiguous(), 
            self._scaling.detach(), 
            self._rotation.detach()], -1).cpu().numpy()
        
        self.depth = int(np.round(np.log2(1/self.voxel_size)) + 2)
        print('self.depth', self.depth)
        # TODO: move to TorchSparse Voxelization.

        
        V, features, oct, paramarr, _, _ = create_octree(
            self._anchor.detach().cpu().numpy(), 
            features,
            imp,
            depth=self.depth,
            oct_merge=merge_type)
        d_anchors, _ = decode_oct(paramarr, oct, self.depth)

        
        if raht:
            # morton sort
            w, val, reorder = copyAsort(V)
            self.reorder = reorder
            self.res = haar3D_param(self.depth, w, val)
            self.res_inv = inv_haar3D_param(V, self.depth)
        
        
        self.oct = oct
        self.oct_param = paramarr
        self._anchor = nn.Parameter(torch.tensor(d_anchors, dtype=torch.float, device="cuda").requires_grad_(True))
        base = 0
        self._anchor_feat = nn.Parameter(torch.tensor(features[:, base:base + self.feat_dim], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        base += self.feat_dim
        self._offset = nn.Parameter(torch.tensor(features[:, base:base + self.n_offsets*3].reshape([-1, self.n_offsets, 3]), dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        base += self.n_offsets*3
        self._opacity = nn.Parameter(torch.tensor(features[:, base:base + 1], dtype=torch.float, device="cuda").requires_grad_(True))
        base += 1
        self._scaling = nn.Parameter(torch.tensor(features[:, base:base+6], dtype=torch.float, device="cuda").requires_grad_(True))
        base += 6
        self._rotation = nn.Parameter(torch.tensor(features[:, base:base+4], dtype=torch.float, device="cuda").requires_grad_(True))
        base += 4
        print('base, feature size', base)
        self.feature_channels = base


    def octree_init_train(self, raht=False):
        features = torch.concat([
            self._anchor_feat.detach(), 
            self._offset.detach().flatten(-2).contiguous(), 
            self._opacity.detach().contiguous(), 
            self._scaling.detach(), 
            self._rotation.detach()], -1).cpu().numpy()
        
        d_anchors = self._anchor.detach().cpu().numpy()

        self._anchor = nn.Parameter(torch.tensor(d_anchors, dtype=torch.float, device="cuda").requires_grad_(True))
        base = 0
        self._anchor_feat = nn.Parameter(torch.tensor(features[:, base:base + self.feat_dim], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        base += self.feat_dim
        self._offset = nn.Parameter(torch.tensor(features[:, base:base + self.n_offsets*3].reshape([-1, self.n_offsets, 3]), dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        base += self.n_offsets*3
        self._opacity = nn.Parameter(torch.tensor(features[:, base:base + 1], dtype=torch.float, device="cuda").requires_grad_(True))
        base += 1
        self._scaling = nn.Parameter(torch.tensor(features[:, base:base+6], dtype=torch.float, device="cuda").requires_grad_(True))
        base += 6
        self._rotation = nn.Parameter(torch.tensor(features[:, base:base+4], dtype=torch.float, device="cuda").requires_grad_(True))
        base += 4
        print('base, feature size', base)
        self.feature_channels = base


    

    def octree_train(self, imp, merge_type, raht=False):

        features = torch.concat([
            self._anchor_feat, 
            self._offset.flatten(-2).contiguous(), 
            self._opacity.contiguous(), 
            self._scaling, 
            self._rotation], -1)
        
        self.depth = int(np.round(np.log2(1/self.voxel_size)) + 2)
        # TODO: move to TorchSparse Voxelization.
        
        self._anchor_v, features_voxel, oct, paramarr, V, octree_imp,  _, _ = create_octree_train(
            self._anchor, 
            features,
            imp,
            pdepht=self.depth,
            oct_merge=merge_type)
        self.octree_imp = octree_imp
             
        if raht:
            # morton sort
            w, val, reorder = copyAsort(V)
            self.reorder = reorder
            self.res = haar3D_param(self.depth, w, val)
            self.res_inv = inv_haar3D_param(V, self.depth)
        
        self.V = V
        self.oct = oct
        self.oct_param = paramarr
        base = 0
        self._anchor_feat_v = features_voxel[:, base:base + self.feat_dim]
        base += self.feat_dim
        self._offset_v = features_voxel[:, base:base + self.n_offsets*3].reshape([-1, self.n_offsets, 3])
        base += self.n_offsets*3
        self._opacity_v = features_voxel[:, base:base + 1]
        base += 1
        self._scaling_v = features_voxel[:, base:base+6]
        base += 6
        self._rotation_v = features_voxel[:, base:base+4]
        base += 4
        self.feature_channels = base

        
        
    def apply_raht_quant(self,training=False):
        # anchor_feat torch.Size([868693, 32]) [20/08 22:10:57]
        # _offset torch.Size([868693, 10, 3]) [20/08 22:10:57]
        # _opacity torch.Size([868693, 1]) [20/08 22:10:57]
        # _scaling torch.Size([868693, 6]) [20/08 22:10:57]
        # _rotation torch.Size([868693, 4]) [20/08 22:10:57] not quant, ok
        if training:
            rf = torch.concat([
            self._anchor_feat_v, #32
            self._offset_v.flatten(-2).contiguous(), # 30
            self._opacity_v.contiguous(), #1
            self._scaling_v, #6
            self._rotation_v], -1) #4
        else:
            rf = torch.concat([
            self._anchor_feat.detach(), #32
            self._offset.detach().flatten(-2).contiguous(), # 30
            self._opacity.detach().contiguous(), #1
            self._scaling.detach(), #6
            self._rotation.detach()], -1) #4


        # t = time.time()
        C = rf[self.reorder]
        iW1 = self.res['iW1']
        iW2 = self.res['iW2']
        iLeft_idx = self.res['iLeft_idx']
        iRight_idx = self.res['iRight_idx']
        
        for d in range(self.depth * 3):
            w1 = iW1[d]
            w2 = iW2[d]
            left_idx = iLeft_idx[d]
            right_idx = iRight_idx[d]
            C[left_idx], C[right_idx] = transform_batched_torch(w1, 
                                                  w2, 
                                                  C[left_idx], 
                                                  C[right_idx])

        # print('forward', time.time()-t, 's')
        # t = time.time()
        
        # np.save('bef_quant.npy', C.detach().cpu().numpy())
        # np.savetxt('bef_quant.csv', C.detach().cpu().numpy(), delimiter=',')

        quantC = torch.zeros_like(C)
        quantC[0] = C[0]
        qa_cnt = 0
        lc1 = C.shape[0] - 1
        # split_ac = split_length(lc1, self.n_block)
        split_ac = self.dp_split
        assert lc1==sum(split_ac[0]), "split length is error"
        scale = torch.zeros(lc1,quantC.shape[1]).to('cuda')
        zero_point = torch.zeros_like(scale)
        thd_neg = torch.zeros_like(scale)
        thd_po = torch.zeros_like(scale)
        for i in range(C.shape[-1]):    
            scale[:, i], zero_point[:, i], thd_neg[:, i], thd_po[:, i] = seg_quant(C[1:, i], split_ac[i], self.qas[qa_cnt : qa_cnt + self.n_block], need_all=True)
            qa_cnt += self.n_block
        quantC[1:] = Quant_all.apply(C[1:], scale, zero_point, thd_neg, thd_po)
        
        # np.save('after_quant.npy', quantC.detach().cpu().numpy())

        # print('quant', time.time()-t, 's')
        # t = time.time()
        
        res_inv = self.res_inv
        pos = res_inv['pos']
        iW1 = res_inv['iW1']
        iW2 = res_inv['iW2']
        iS = res_inv['iS']
        
        iLeft_idx = res_inv['iLeft_idx']
        iRight_idx = res_inv['iRight_idx']
    
        iLeft_idx_CT = res_inv['iLeft_idx_CT']
        iRight_idx_CT = res_inv['iRight_idx_CT']
        iTrans_idx = res_inv['iTrans_idx']
        iTrans_idx_CT = res_inv['iTrans_idx_CT'] 

        CT_yuv_q_temp = quantC[pos.astype(int)]
        drf = torch.zeros(quantC.shape).cuda()
        OC = torch.zeros(quantC.shape).cuda()
        
        for i in range(self.depth*3):
            w1 = iW1[i]
            w2 = iW2[i]
            S = iS[i]
            
            left_idx, right_idx = iLeft_idx[i], iRight_idx[i]
            left_idx_CT, right_idx_CT = iLeft_idx_CT[i], iRight_idx_CT[i]
            
            trans_idx, trans_idx_CT = iTrans_idx[i], iTrans_idx_CT[i]
            
            
            OC[trans_idx] = CT_yuv_q_temp[trans_idx_CT]
            OC[left_idx], OC[right_idx] = itransform_batched_torch(w1, 
                                                    w2, 
                                                    CT_yuv_q_temp[left_idx_CT], 
                                                    CT_yuv_q_temp[right_idx_CT])  
            CT_yuv_q_temp[:S] = OC[:S]

        drf[self.reorder] = OC
        ddrf = torch.mean(torch.square(drf.detach()-rf.detach()), dim=0)
        # print('backward', time.time()-t, 's')
        
        # np.save('ddrf.npy', ddrf.cpu().numpy())
        # np.save('drf.npy', drf.detach().cpu().numpy())
        return drf

    def apply_quant(self):
        rf = torch.concat([
            self._anchor_feat.detach(), 
            self._offset.detach().flatten(-2).contiguous(), 
            self._opacity.detach().contiguous(), 
            self._scaling.detach(), 
            self._rotation.detach()], -1)

        qa_cnt = 0
        lc1 = rf.shape[0]
        drf = torch.zeros_like(rf)
        # split_ac = split_length(lc1, self.n_block)
        split_ac = self.dp_split
        assert lc1==sum(split_ac[0]), "split length is error"
        for i in range(rf.shape[-1]):
            drf[:, i] = seg_quant(rf[:, i], split_ac[i], self.qas[qa_cnt : qa_cnt + self.n_block])
            qa_cnt += self.n_block
        
        return drf

    def set_dp_split(self, dp=True, unit_length=100, n_block=None):
        self.n_block = n_block
        if dp==False:
            self.dp_split = [split_length(self._anchor_v.shape[0]-1, self.n_block)]*73
            return
        rf = torch.concat([
            self._anchor_feat_v.detach(), 
            self._offset_v.detach().flatten(-2).contiguous(), 
            self._opacity_v.detach().contiguous(), 
            self._scaling_v.detach(), 
            self._rotation_v.detach()], -1)

        C = rf[self.reorder]
        iW1 = self.res['iW1']
        iW2 = self.res['iW2']
        iLeft_idx = self.res['iLeft_idx']
        iRight_idx = self.res['iRight_idx']
        
        for d in range(self.depth * 3):
            w1 = iW1[d]
            w2 = iW2[d]
            left_idx = iLeft_idx[d]
            right_idx = iRight_idx[d]
            C[left_idx], C[right_idx] = transform_batched_torch(w1, 
                                                  w2, 
                                                  C[left_idx], 
                                                  C[right_idx])

        split_0 = dp_split(C[1:, :32], self.qas[:32*self.n_block], self.n_block, C.shape[0]-1, unit_length)
        print("Finished_1")
        # split_1 = dp_split(C[1:, 32:], self.qas[32*self.n_block:], self.n_block, C.shape[0]-1)
        print("Finished_2")
        # split = [split_0] * 32 + [split_1] * 41
        split = [split_0] * 73
        self.dp_split = split

    def change_dp_split(self, dp=True, unit_length=100):
        if self._anchor_v.shape[0]-1 == sum(self.dp_split[0]):
            # print("anchor",self._anchor_v.shape[0]-1,"dp_split",sum(self.dp_split[0]))
            return
        if dp==False:
            self.dp_split = [split_length(self._anchor_v.shape[0]-1, self.n_block)]*73
            return
        rf = torch.concat([
            self._anchor_feat_v.detach(), 
            self._offset_v.detach().flatten(-2).contiguous(), 
            self._opacity_v.detach().contiguous(), 
            self._scaling_v.detach(), 
            self._rotation_v.detach()], -1)

        C = rf[self.reorder]
        iW1 = self.res['iW1']
        iW2 = self.res['iW2']
        iLeft_idx = self.res['iLeft_idx']
        iRight_idx = self.res['iRight_idx']
        
        for d in range(self.depth * 3):
            w1 = iW1[d]
            w2 = iW2[d]
            left_idx = iLeft_idx[d]
            right_idx = iRight_idx[d]
            C[left_idx], C[right_idx] = transform_batched_torch(w1, 
                                                  w2, 
                                                  C[left_idx], 
                                                  C[right_idx])

        split_0 = dp_split(C[1:, :32], self.qas[:32*self.n_block], self.n_block, C.shape[0]-1, unit_length)
        split = [split_0] * 73
        self.dp_split = split

    
    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1) # 出现次数少的也别grow了
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1) # opacity_accum又太少了
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1] anchor出现次数过大的
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError

    def save(self):
        means3D = self.get_anchor_v
        scales = self.get_scaling_v
        rotations = self.get_rotation_v
        tensors_dict = {'means3D': means3D, 'scales': scales[:,:3], 'rotations': rotations}
        torch.save(tensors_dict, 'tensors_v.pth')
        means3D = self.get_anchor
        scales = self.get_scaling
        rotations = self.get_rotation
        tensors_dict = {'means3D': means3D, 'scales': scales[:,:3], 'rotations': rotations}
        torch.save(tensors_dict, 'tensors.pth')

    def test_grad(self):
        self._anchor_v.retain_grad()
        x = self._anchor_v**2 +self._anchor_v
        loss = x.sum()
        loss.backward()

        print('test!!!!!')
        print(self._anchor_v.grad)
        print(self._anchor.grad)

    def octree_init(self):
        self._anchor_v = self._anchor
        self._anchor_v.retain_grad()
        self._anchor_feat_v = self._anchor_feat
        self._offset_v = self._offset
        self._opacity_v = self._opacity
        self._scaling_v = self._scaling
        self._rotation_v = self._rotation

    def get_ddrf(self):

        rf = torch.concat([
            self._anchor_feat_v, #32
            self._offset_v.flatten(-2).contiguous(), # 30
            self._opacity_v.contiguous(), #1
            self._scaling_v, #6
            self._rotation_v], -1) #4


        # t = time.time()
        C = rf[self.reorder]
        iW1 = self.res['iW1']
        iW2 = self.res['iW2']
        iLeft_idx = self.res['iLeft_idx']
        iRight_idx = self.res['iRight_idx']
        
        for d in range(self.depth * 3):
            w1 = iW1[d]
            w2 = iW2[d]
            left_idx = iLeft_idx[d]
            right_idx = iRight_idx[d]
            C[left_idx], C[right_idx] = transform_batched_torch(w1, 
                                                  w2, 
                                                  C[left_idx], 
                                                  C[right_idx])


        quantC = torch.zeros_like(C)
        quantC[0] = C[0]
        qa_cnt = 0
        lc1 = C.shape[0] - 1
        split_ac = self.dp_split
        assert lc1==sum(split_ac[0]), "split length is error"

        A = torch.zeros(C.shape[-1], 16).cuda()

        
        for i in range(C.shape[-1]):   
            for j in range(1,17): 
                qas_temp = nn.ModuleList([])
                for k in range(self.n_block):
                    qas_temp.append(VanillaQuan(bit=j).cuda())    
                quant = seg_quant(C[1:, i], split_ac[i], qas_temp, need_all=False)
                A[i,j-1] = torch.sum(torch.square(quant.detach()-C[1:, i].detach()))
        
        A_flat = A.flatten().cpu().tolist()  # 将展平后的张量转为 Python 列表

        return A_flat

        
