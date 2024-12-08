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
import pickle
from einops import repeat
import numpy as np
import math
import json
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, meson_count=False, save=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training or meson_count:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, meson_count=False, show=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training or meson_count:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, meson_count=meson_count)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, meson_count=meson_count)
    

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        meson_count=meson_count,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    torch.cuda.synchronize()
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if raster_settings.meson_count:
        rendered_image, radii, imp = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = color,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)
    else:
        rendered_image, radii = rasterizer(
            means3D = xyz,   #important
            means2D = screenspace_points,
            shs = None,
            colors_precomp = color,
            opacities = opacity,
            scales = scaling,
            rotations = rot,   #important
            cov3D_precomp = None)
    torch.cuda.empty_cache()
        
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    elif meson_count:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "imp": imp,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }

def rendersave(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, meson_count=False, show=False, idx=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training or meson_count:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, meson_count=meson_count)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, meson_count=meson_count)
    

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        meson_count=meson_count,
        debug=pipe.debug
    )

    data_to_save = {
        'image_height': int(viewpoint_camera.image_height),
        'image_width': int(viewpoint_camera.image_width),
        'tanfovx': tanfovx,
        'tanfovy': tanfovy,
        'bg': bg_color,
        'scale_modifier': scaling_modifier,
        'viewmatrix': viewpoint_camera.world_view_transform,
        'projmatrix': viewpoint_camera.full_proj_transform,
        'sh_degree': 1,
        'campos': viewpoint_camera.camera_center,
        'prefiltered': False,
        'meson_count': meson_count,
        'debug': pipe.debug
    }

    # 保存到文件
    with open('renderdata/data{}.pkl'.format(idx), 'wb') as f:
        pickle.dump(data_to_save, f)

    data_to_save = {
        'means3D': xyz,  # important
        'means2D': screenspace_points,
        'shs': None,
        'colors_precomp': color,
        'opacities': opacity,
        'scales': scaling,
        'rotations': rot,  # important
        'cov3D_precomp': None
    }

    # 保存到文件
    torch.save(data_to_save, 'renderdata/tensor_data{}.pt'.format(idx))


def renderload(idx=-1):
    with open('renderdata/data{}.pkl'.format(idx), 'rb') as f:
        loaded_data = pickle.load(f)

    # idx = 34
    # 提取加载的数据
    image_height = loaded_data['image_height']
    image_width = loaded_data['image_width']
    tanfovx = loaded_data['tanfovx']
    tanfovy = loaded_data['tanfovy']
    bg_color = loaded_data['bg']
    scaling_modifier = loaded_data['scale_modifier']
    viewmatrix = loaded_data['viewmatrix']  # 假设这是一个矩阵或数组
    projmatrix = loaded_data['projmatrix']  # 同上
    sh_degree = loaded_data['sh_degree']
    campos = loaded_data['campos']  # 假设这是一个数组
    prefiltered = loaded_data['prefiltered']
    meson_count = loaded_data['meson_count']
    debug = loaded_data['debug']


    # 从文件中加载数据
    loaded_data = torch.load('renderdata/tensor_data{}.pt'.format(idx))

    # 访问加载的数据
    xyz = loaded_data['means3D']
    screenspace_points = loaded_data['means2D']
    color = loaded_data['colors_precomp']
    opacity = loaded_data['opacities']
    scaling = loaded_data['scales']
    rot = loaded_data['rotations']  # important


    raster_settings = GaussianRasterizationSettings(
            image_height=int(image_height),
            image_width=int(image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=1,
            campos=campos,
            prefiltered=False,
            meson_count=meson_count,
            debug=debug
        )


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii = rasterizer(
        means3D = xyz,   #important
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,   #important
        cov3D_precomp = None)

    torch.cuda.empty_cache()
    return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }
    

def ft_gng(viewpoint_camera, rf, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor_v.shape[0], dtype=torch.bool, device = pc.get_anchor_v.device)
    
    base = 0
    feat = rf[:, base:base+pc.feat_dim][visible_mask]
    base += pc.feat_dim
    anchor = pc.get_anchor_v[visible_mask]
    grid_offsets = rf[:, base:base + pc.n_offsets*3].reshape([-1, pc.n_offsets, 3])[visible_mask]
    base += pc.n_offsets*3
    base += 1
    grid_scaling = rf[:, base:base + 6][visible_mask]
    grid_scaling = 1.0*pc.scaling_activation(grid_scaling)
    base+=6

    # TODO: check if HAC used prefilter during rendering?
    # np.savetxt('feat_ftrender.csv', feat.detach().cpu().numpy(), delimiter=',')
    # np.savetxt('offsets_ftrender.csv', grid_offsets.detach().cpu().numpy().reshape(grid_offsets.shape[0], -1), delimiter=',')
    # np.savetxt('scaling_ftrender.csv', grid_scaling.detach().cpu().numpy().reshape(grid_scaling.shape[0], -1), delimiter=',')

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

def ft_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, meson_count=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = True
    
    if pc.raht:
        drf = pc.apply_raht_quant(training=True)
    else:
        drf = pc.apply_quant()

    xyz, color, opacity, scaling, rot, neural_opacity, mask = ft_gng(viewpoint_camera, drf, pc, visible_mask, is_training=is_training)

    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor_v.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        meson_count=meson_count,
        debug=pipe.debug
    )

    raster_settings_dict = {
        "image_height": int(viewpoint_camera.image_height),
        "image_width": int(viewpoint_camera.image_width),
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "scale_modifier": scaling_modifier,
        "sh_degree": 1,
        "prefiltered": False,
        "meson_count": meson_count,
        "debug": pipe.debug
    }


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if raster_settings.meson_count:
        rendered_image, radii, imp = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = color,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)
    else:
        rendered_image, radii = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = color,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3D_precomp = None)
        
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    elif meson_count:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "imp": imp,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, training=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if training:
        screenspace_points = torch.zeros_like(pc.get_anchor_v, dtype=pc.get_anchor_v.dtype, requires_grad=True, device="cuda") + 0
    else:
        screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        meson_count=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if training:
        means3D = pc.get_anchor_v
    else:
        means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if training:
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance_v(scaling_modifier)
        else:
            scales = pc.get_scaling_v
            rotations = pc.get_rotation_v
    else:
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

    

    radii_pure = rasterizer.visible_filter(means3D = means3D, #(N,3)
        scales = scales[:,:3], #(N,6)
        rotations = rotations, #(N,4)
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
