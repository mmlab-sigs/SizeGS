import numpy as np 
import torch
import octree_cuda
import time
from typing import Union
from plyfile import PlyData
import os

class UniqueWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ppoints, occode, letra, indices):
        koorx=letra[0][occode[0]]
        koory=letra[1][occode[1]]
        koorz=letra[2][occode[2]]
        voxel_xyz = torch.stack([koorx, koory, koorz], dim=-1)
        ctx.save_for_backward(indices, torch.tensor(
            ppoints.shape, dtype=torch.int))

        result = voxel_xyz.clone().detach().requires_grad_(True).to('cuda')
        return result

    @staticmethod
    def backward(ctx, grad_output):
        indices, size = ctx.saved_tensors
        size = size.tolist()
        out_grad = torch.zeros(size, dtype=grad_output.dtype, device='cuda')
        for i in range(len(indices)-1):
            start = indices[i]
            end = indices[i+1]
            out_grad[start:end, :] = grad_output[i, :]
        return out_grad, None, None, None


class Octree_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ppoints, occodex, occodey, occodez, letrax, letray, letraz, indices):
        feat_interp = octree_cuda.octree_fw(occodex, occodey, occodez, letrax, letray, letraz, indices)

        ctx.save_for_backward(indices, ppoints)

        return feat_interp

    @staticmethod
    def backward(ctx, dL_danchor_v):
        indices, ppoints = ctx.saved_tensors

        dL_danchor = octree_cuda.octree_bw(dL_danchor_v.contiguous(), indices, ppoints)

        return dL_danchor, None, None, None, None, None, None, None


class Octree_feature_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pfeatures, indices, imp, oct_merge):
        feature = octree_cuda.feature_fw(pfeatures, indices, imp, oct_merge)

        if oct_merge == 'mean':
            ctx.save_for_backward(pfeatures, indices, None)
        elif oct_merge == 'imp':
            ctx.save_for_backward(pfeatures, indices, imp)
        else:
            raise("Error")

        return feature

    @staticmethod
    def backward(ctx, dL_dfeature_v):
        pfeatures, indices, imp = ctx.saved_tensors

        if imp != None:
            dL_dfeature = octree_cuda.feature_bw(dL_dfeature_v, pfeatures, indices, imp, 'imp')
        else:
            dL_dfeature = octree_cuda.feature_bw(dL_dfeature_v, pfeatures, indices, torch.zeros(pfeatures.shape[0], device='cuda'), 'mean')

        return dL_dfeature, None, None, None


def d1halfing_fast(pmin,pmax,pdepht):
    return np.linspace(pmin,pmax,2**int(pdepht)+1)

def d1halfing_fast_torch(pmin, pmax, pdepht):
    return torch.linspace(pmin, pmax, steps=2**int(pdepht) + 1).to('cuda')


def create_octree_train(ppoints, pfeatures, imp, pdepht, oct_merge, use_pcc=False):
    ori_points_num = ppoints.shape[0]
    # t = time.time()
    torch.cuda.synchronize()  # 在可能出错的CUDA操作之前调用
    imp = imp[:ori_points_num]
    minx = torch.min(ppoints[:, 0]).item()
    maxx = torch.max(ppoints[:, 0]).item()
    miny = torch.min(ppoints[:, 1]).item()
    maxy = torch.max(ppoints[:, 1]).item()
    minz = torch.min(ppoints[:, 2]).item()
    maxz = torch.max(ppoints[:, 2]).item()
    xletra = d1halfing_fast_torch(minx, maxx, pdepht)
    yletra = d1halfing_fast_torch(miny, maxy, pdepht)
    zletra = d1halfing_fast_torch(minz, maxz, pdepht)
    otcodex = torch.searchsorted(xletra, ppoints[:, 0].contiguous(), right=True) - 1
    otcodey = torch.searchsorted(yletra, ppoints[:, 1].contiguous(), right=True) - 1
    otcodez = torch.searchsorted(zletra, ppoints[:, 2].contiguous(), right=True) - 1
    ki = otcodex * (2**(pdepht * 2)) + otcodey * (2**pdepht) + otcodez

    ki_ranks = torch.argsort(ki)
    
    # 根据排序的结果重新排列点和特征
    ppoints = ppoints[ki_ranks]
    pfeatures = pfeatures[ki_ranks]
    ki = ki[ki_ranks]

    ki, counts = torch.unique(ki, return_counts=True)
    imp = imp.to('cuda')

    indices = torch.cumsum(torch.cat((torch.tensor([0], device='cuda'), counts[:-1])), dim=0)
    octree_imp = imp[indices]
    indices = torch.cat((indices, torch.tensor([ki.shape[0]], device='cuda')))

    occodex = (ki / (2**(pdepht * 2))).long()
    occodey = ((ki - occodex * (2**(pdepht * 2))) / (2**pdepht)).long()
    occodez = (ki - occodex * (2**(pdepht * 2)) - occodey * (2**pdepht)).long()
    # print('prepare',time.time()-t,'s')
    # t = time.time()

    d_xyz = Octree_py.apply(ppoints, occodex, occodey, occodez, xletra, yletra, zletra, indices)

    # print('forward',time.time()-t,'s')
    # t = time.time()
    final_points_num = ppoints.shape[0]

    features = Octree_feature_py.apply(pfeatures, indices, imp.to('cuda'), oct_merge)

    # torch_dict = {'pfeatrues':pfeatures,  'features':features, 'indices':indices, 'imp':imp}
    # torch.save(torch_dict, 'feature.pth')

    # print('featrue',time.time()-t,'s')
    # t = time.time()
    # depth and boundary
    paramarr = np.asarray([minx, maxx, miny, maxy, minz, maxz]) 
    occodex_np = occodex.cpu().numpy()
    occodey_np = occodey.cpu().numpy()
    occodez_np = occodez.cpu().numpy()

    # 堆叠 NumPy 数组
    V = np.vstack([occodex_np, occodey_np, occodez_np]).T
    
    if use_pcc:
        V, features, indices_sorted = sorted_voxels(V, features)
        
    d_xyz = d_xyz[indices_sorted]
    
    # V = np.array([occodex.cpu(),occodey.cpu(),occodez.cpu()], dtype=int).T
    # V = torch.stack([occodex.cpu(), occodey.cpu(), occodez.cpu()], dim=1).numpy()
    ki = ki.detach().cpu().numpy()

    # print('retrun',time.time()-t,'s')
    # t = time.time()
    #!!!!!!!!!!!
    return d_xyz, features, ki, paramarr, V, octree_imp, ori_points_num, final_points_num



def octreecodes(ppoints, pdepht, merge_type='mean',imps=None):
    minx=np.amin(ppoints[:,0])
    maxx=np.amax(ppoints[:,0])
    miny=np.amin(ppoints[:,1])
    maxy=np.amax(ppoints[:,1])
    minz=np.amin(ppoints[:,2])
    maxz=np.amax(ppoints[:,2])
    xletra=d1halfing_fast(minx,maxx,pdepht)
    yletra=d1halfing_fast(miny,maxy,pdepht)
    zletra=d1halfing_fast(minz,maxz,pdepht)
    otcodex=np.searchsorted(xletra,ppoints[:,0],side='right')-1
    otcodey=np.searchsorted(yletra,ppoints[:,1],side='right')-1
    otcodez=np.searchsorted(zletra,ppoints[:,2],side='right')-1
    ki=otcodex*(2**(pdepht*2))+otcodey*(2**pdepht)+otcodez
    
    ki_ranks = np.argsort(ki)
    ppoints = ppoints[ki_ranks]
    ki = ki[ki_ranks]

    ppoints = np.concatenate([ki.reshape(-1, 1), ppoints], -1)
    dedup_points = np.split(ppoints[:, 1:], np.unique(ki, return_index=True)[1][1:])
    
    final_feature = []
    if merge_type == 'mean':
        for dedup_point in dedup_points:
            # print(np.mean(dedup_point, 0).shape)
            final_feature.append(np.mean(dedup_point, 0).reshape(1, -1))
    elif merge_type == 'imp':
        dedup_imps = np.split(imps, np.unique(ki, return_index=True)[1][1:])
        for dedup_point, dedup_imp in zip(dedup_points, dedup_imps):
            dedup_imp = dedup_imp.reshape(1, -1)
            if dedup_imp.shape[-1] == 1:
                # print('dedup_point.shape', dedup_point.shape)
                final_feature.append(dedup_point)
            else:
                # print('dedup_point.shape, dedup_imp.shape', dedup_point.shape, dedup_imp.shape)
                fdp = (dedup_imp / np.sum(dedup_imp)) @ dedup_point
                # print('fdp.shape', fdp.shape)
                final_feature.append(fdp)
    elif merge_type == 'rand':
        for dedup_point in dedup_points:
            ld = len(dedup_point)
            id = torch.randint(0, ld, (1,))[0]
            final_feature.append(dedup_point[id].reshape(1, -1))
    else:
        raise NotImplementedError
    ki = np.unique(ki)
    final_feature = np.concatenate(final_feature, 0)
    # print('final_feature.shape', final_feature.shape)
    return (ki,minx,maxx,miny,maxy,minz,maxz, final_feature)

def create_octree(ppoints, pfeatures, imp, depth, oct_merge):
    ori_points_num = ppoints.shape[0]
    ppoints = np.concatenate([ppoints, pfeatures], -1)
    occ=octreecodes(ppoints, depth, oct_merge, imp)
    final_points_num = occ[0].shape[0]
    occodex=(occ[0]/(2**(depth*2))).astype(int)
    occodey=((occ[0]-occodex*(2**(depth*2)))/(2**depth)).astype(int)
    occodez=(occ[0]-occodex*(2**(depth*2))-occodey*(2**depth)).astype(int)
    voxel_xyz = np.array([occodex,occodey,occodez], dtype=int).T
    features = occ[-1][:, 3:]
    paramarr=np.asarray([occ[1],occ[2],occ[3],occ[4],occ[5],occ[6]]) # boundary
    # print('oct[0]', type(oct[0]))
    return voxel_xyz, features, occ[0], paramarr, ori_points_num, final_points_num

def decode_oct(paramarr, oct, depth):
    minx=(paramarr[0])
    maxx=(paramarr[1])
    miny=(paramarr[2])
    maxy=(paramarr[3])
    minz=(paramarr[4])
    maxz=(paramarr[5])
    xletra=d1halfing_fast(minx,maxx,depth)
    yletra=d1halfing_fast(miny,maxy,depth)
    zletra=d1halfing_fast(minz,maxz,depth)
    occodex=(oct/(2**(depth*2))).astype(int)
    occodey=((oct-occodex*(2**(depth*2)))/(2**depth)).astype(int)
    occodez=(oct-occodex*(2**(depth*2))-occodey*(2**depth)).astype(int)  
    V = np.array([occodex,occodey,occodez], dtype=int).T # [P, 3]
    koorx=xletra[occodex]
    koory=yletra[occodey]
    koorz=zletra[occodez]
    ori_points=np.array([koorx,koory,koorz]).T

    return ori_points, V

def decode_v(paramarr, V, depth):
    minx=(paramarr[0])
    maxx=(paramarr[1])
    miny=(paramarr[2])
    maxy=(paramarr[3])
    minz=(paramarr[4])
    maxz=(paramarr[5])
    xletra=d1halfing_fast(minx,maxx,depth)
    yletra=d1halfing_fast(miny,maxy,depth)
    zletra=d1halfing_fast(minz,maxz,depth)
    # occodex=(oct/(2**(depth*2))).astype(int)
    # occodey=((oct-occodex*(2**(depth*2)))/(2**depth)).astype(int)
    # occodez=(oct-occodex*(2**(depth*2))-occodey*(2**depth)).astype(int)  
    # V = np.array([occodex,occodey,occodez], dtype=int).T
    V = V.astype(int)
    assert V.shape[-1] == 3
    koorx=xletra[V[:, 0].reshape(-1)]
    koory=yletra[V[:, 1].reshape(-1)]
    koorz=zletra[V[:, 2].reshape(-1)]
    ori_points=np.array([koorx,koory,koorz]).T

    return ori_points, V


def sorted_voxels(voxelized_means: np.ndarray, other_params = None) -> Union[np.ndarray, tuple]:
    """
    Sort voxels by their Morton code.
    """
    indices_sorted = np.argsort(voxelized_means @ np.power(voxelized_means.max() + 1, np.arange(voxelized_means.shape[1])), axis=0)
    voxelized_means = voxelized_means[indices_sorted]
    if other_params is None:
        return voxelized_means
    other_params = other_params[indices_sorted]
    return voxelized_means, other_params, indices_sorted


def gpcc_encode(ply_path: str, bin_path: str, encoder_path='/home/szxie/storage/mpeg-pcc-tmc13/build/tmc3/tmc3') -> None:
    """
    Compress geometry point cloud by GPCC codec.
    """
    enc_cmd = (f'{encoder_path} '
               f'--mode=0 --trisoupNodeSizeLog2=0 --mergeDuplicatedPoints=0 --neighbourAvailBoundaryLog2=8 '
               f'--intra_pred_max_node_size_log2=3 --positionQuantizationScale=1 --inferredDirectCodingMode=3 '
               f'--maxNumQtBtBeforeOt=2 --minQtbtSizeLog2=0 --planarEnabled=0 --planarModeIdcmUse=0 --cabac_bypass_stream_enabled_flag=1 '
               f'--uncompressedDataPath={ply_path} --compressedStreamPath={bin_path} ')
    enc_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    exit_code = os.system(enc_cmd)
    assert exit_code == 0, f'GPCC encoder failed with exit code {exit_code}.'


def gpcc_decode(bin_path: str, recon_path: str, decoder_path='/home/szxie/storage/mpeg-pcc-tmc13/build/tmc3/tmc3') -> None:
    """
    Decompress geometry point cloud by GPCC codec.
    """
    dec_cmd = (f'{decoder_path} '
               f'--mode=1 --outputBinaryPly=1 '
               f'--compressedStreamPath={bin_path} --reconstructedDataPath={recon_path} ')
    dec_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    exit_code = os.system(dec_cmd)
    assert exit_code == 0, f'GPCC decoder failed with exit code {exit_code}.'


def write_ply_geo_ascii(geo_data: np.ndarray, ply_path: str) -> None:
    """
    Write geometry point cloud to a .ply file in ASCII format.
    """
    assert ply_path.endswith('.ply'), 'Destination path must be a .ply file.'
    assert geo_data.ndim == 2 and geo_data.shape[1] == 3, 'Input data must be a 3D point cloud.'
    geo_data = geo_data.astype(int)
    with open(ply_path, 'w') as f:
        # write header
        f.writelines(['ply\n', 'format ascii 1.0\n', f'element vertex {geo_data.shape[0]}\n',
                      'property float x\n', 'property float y\n', 'property float z\n', 'end_header\n'])
        # write data
        for point in geo_data:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')


def read_ply_geo_bin(ply_path: str) -> np.ndarray:
    """
    Read geometry point cloud from a .ply file in binary format.
    """
    assert ply_path.endswith('.ply'), 'Source path must be a .ply file.'

    ply_data = PlyData.read(ply_path).elements[0]
    means = np.stack([ply_data.data[name] for name in ['x', 'y', 'z']], axis=1)  # shape (N, 3)
    return means

# def sorted_orig_voxels(voxelized_means, other_params=None):
#     # means = means.detach().cpu().numpy().astype(np.float32)
#     # voxelized_means, means_min, means_max = voxelize(means=means)
    
#     voxelized_means, other_params = sorted_voxels(voxelized_means=voxelized_means, other_params=other_params)
#     means = devoxelize(voxelized_means=voxelized_means, means_min=means_min, means_max=means_max)
#     means = torch.from_numpy(means).cuda().to(torch.float32)
#     return means, other_params


# def create_octree_train_test(ppoints, pfeatures, imp, pdepht, oct_merge):
#     ori_points_num = ppoints.shape[0]
#     # t = time.time()
#     imp = imp[:ori_points_num]
#     minx = torch.min(ppoints[:, 0]).item()
#     maxx = torch.max(ppoints[:, 0]).item()
#     miny = torch.min(ppoints[:, 1]).item()
#     maxy = torch.max(ppoints[:, 1]).item()
#     minz = torch.min(ppoints[:, 2]).item()
#     maxz = torch.max(ppoints[:, 2]).item()
#     xletra = d1halfing_fast_torch(minx, maxx, pdepht)
#     yletra = d1halfing_fast_torch(miny, maxy, pdepht)
#     zletra = d1halfing_fast_torch(minz, maxz, pdepht)
#     otcodex = torch.searchsorted(xletra, ppoints[:, 0].contiguous(), right=True) - 1
#     otcodey = torch.searchsorted(yletra, ppoints[:, 1].contiguous(), right=True) - 1
#     otcodez = torch.searchsorted(zletra, ppoints[:, 2].contiguous(), right=True) - 1
#     ki = otcodex * (2**(pdepht * 2)) + otcodey * (2**pdepht) + otcodez

#     ki_ranks = torch.argsort(ki)
    
#     # 根据排序的结果重新排列点和特征
#     ppoints = ppoints[ki_ranks]
#     pfeatures = pfeatures[ki_ranks]
#     ki = ki[ki_ranks]

#     ki, counts = torch.unique(ki, return_counts=True)
#     imp = imp.to('cuda')

#     indices = torch.cumsum(torch.cat((torch.tensor([0], device='cuda'), counts[:-1])), dim=0)
#     octree_imp = imp[indices]
#     indices = torch.cat((indices, torch.tensor([ki.shape[0]], device='cuda')))

#     occodex = (ki / (2**(pdepht * 2))).long()
#     occodey = ((ki - occodex * (2**(pdepht * 2))) / (2**pdepht)).long()
#     occodez = (ki - occodex * (2**(pdepht * 2)) - occodey * (2**pdepht)).long()

#     d_xyz = Octree_py.apply(ppoints, occodex, occodey, occodez, xletra, yletra, zletra, indices)

#     final_points_num = d_xyz.shape[0]

#     features = Octree_feature_py.apply(pfeatures, indices, imp, oct_merge)
#     paramarr = np.asarray([minx, maxx, miny, maxy, minz, maxz]) 
#     V = np.array([occodex.cpu(),occodey.cpu(),occodez.cpu()], dtype=int).T
#     ki = ki.detach().cpu().numpy()

#     return d_xyz, features, ki, paramarr, V, octree_imp, ori_points_num, final_points_num