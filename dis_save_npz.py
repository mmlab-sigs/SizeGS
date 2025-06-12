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
import os
import torch

import numpy as np
# import mediapy as media
from matplotlib import cm
from PIL import Image
from render_utils import generate_path, create_videos
from utils.image_utils import psnr
from utils.quant_utils import split_length
import subprocess
from multiprocessing import Pool
from utils.loss_utils import l1_loss, ssim

from scene import Scene
from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, LpStatus, PULP_CBC_CMD
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import torch.nn.functional as F 
import lpips
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

def per_channel_qbit_upper(cha_qbits, qbit_size_limit):
    qbitsum = sum(cha_qbits)
    elem = qbit_size_limit / qbitsum 
    per_channel_uppers = []
    for bi in cha_qbits:
        per_channel_uppers.append(bi * elem)
    return per_channel_uppers

def qbit_block_size_esimator(qbits, n_bit, low_bit, n_block, splits, use_entropy):
    # qbits: [0 or 1]
    if use_entropy:
        pass
    else:
        qbsz = 0
        for i, si in enumerate(splits):
            for j in range(n_bit):
                qbsz += si * qbits[i*n_bit + j] * (j + low_bit)
    return qbsz # bit

def solve_channel_ilp_pulp(L, n_bit, low_bit, n_block, splits, model_size_limit, use_entropy: bool, time_limit, warmup=False, vals=None):
    num_variable = n_block * n_bit 
    variable = {}
    for i in range(num_variable):
        variable[f"x{i}"] = LpVariable(f"x{i}", 0, 1, cat=LpInteger)
    prob = LpProblem("Model_Size", LpMinimize)

    prob += qbit_block_size_esimator([variable[f"x{i}"] for i in range(num_variable)], n_bit, low_bit, n_block, splits, use_entropy) <= model_size_limit

    for le in range(0, n_block):
        ri = le * n_bit + n_bit
        prob += sum([variable[f"x{i}"] for i in range(le * n_bit, ri)]) == 1

    prob += sum([(variable[f"x{i}"]) * L[i] for i in range(num_variable)])

    if warmup:
        assert vals is not None 
        for i in range(num_variable):
            if vals[i] == 0 or vals[i] == 1:
                variable[f"x{i}"].setInitialValue(vals[i])
            else:
                variable[f"x{i}"].setInitialValue(0)

    # 使用CBC求解器并设置时间限制
    prob.solve(PULP_CBC_CMD(timeLimit=time_limit, msg=True))

    solution = {}
    for i in range(num_variable):
        solution[f"x{i}"] = variable[f"x{i}"].varValue

    last_vals = []
    qbits = [0]*n_block
    for i in range(n_block):
        for j in range(n_bit):
            last_vals.append(int(variable[f"x{i*n_bit+j}"].varValue))
            if variable[f"x{i*n_bit+j}"].varValue != 0:
                qbits[i] = j+low_bit 
    
    return qbits, last_vals

def final_qbit_size_estimator(qbits, splits, n_channel, n_block, enable_drop: bool):
    if enable_drop:
        n_c = n_channel - 3
    else:
        n_c = n_channel
    qbsz = 0
    for i in range(n_c):
        for j in range(n_block):
            qbsz += qbits[i, j] * splits[i, j]
    return qbsz

# 将每个通道的处理封装为函数
def solve_wrapper(args):
    L, n_bit, low_bit, n_block, splits, model_size_limit, use_entropy, time_limit, warmup, vals = args
    return solve_channel_ilp_pulp(
        L=L,
        n_bit=n_bit,
        low_bit=low_bit,
        n_block=n_block,
        splits=splits,
        model_size_limit=model_size_limit,
        use_entropy=use_entropy,
        time_limit=time_limit,
        warmup=warmup,
        vals=vals
    )
    
def research_fine(
    A,
    former_qbits,
    gaussians,
    scene,
    model_size_limit,
    bit_lower_bound,
    n_bit,
    n_block,
    enable_drop,
    split,
    use_entropy=False
):
    warmup = True 
    gaussians.init_bqa(former_qbits)
    # logger.info('cha_qbits {}'.format(cha_qbits))
    cha_qbits_one_hot = F.one_hot(former_qbits - bit_lower_bound, num_classes=n_bit)
    cha_qbits_one_hot = cha_qbits_one_hot.tolist()
    # logger.info('cha_qbits_one_hot {} '.format(cha_qbits_one_hot))
    last_vals = [iqb_one_hot*n_block for iqb_one_hot in cha_qbits_one_hot]
    # logger.info('len(last_vals) {}, len(last_vals[0]) {}'.format(len(last_vals), len(last_vals[0])))
    time_limit = 50
    actual_sizes =[]
    qbit_size_limits = []
    for j in range(4):
        per_channel_size_limit = per_channel_qbit_upper(former_qbits, qbit_size_limit)
        if enable_drop:
            assert len(per_channel_size_limit) == 70
                
        args_list = [
            (
                A[i].flatten().cpu().tolist(),  # 提前转换为CPU列表
                n_bit,
                bit_lower_bound,
                n_block,
                split[i].reshape(-1).cpu().tolist(),
                pc_szlmit,
                use_entropy, 
                time_limit,
                warmup,
                last_vals[i]
            )
            for i, pc_szlmit in enumerate(per_channel_size_limit)
        ]

        with Pool() as pool:  # 自动使用所有CPU核心
            results = pool.map(solve_wrapper, args_list)
        
        final_qbits = [torch.tensor(res[0], device="cuda", dtype=int) for res in results] 
        last_vals = [res[1] for res in results]     
        # logger.info("last_vals {}".format(last_vals))       
        final_qbits = torch.stack(final_qbits)
        # logger.info("final_qbits {}".format(final_qbits))  
        logger.info(f"{j} qbit_size_limit: {qbit_size_limit / 8 } bytes")
        
        warmup=True
        gaussians.update_bqa_wo_refresh(qbits=final_qbits)
        actual_size = scene.save_compressed(0) * 8 # bits
        
        if j > 0:
            delta_actual_size = actual_size - actual_sizes[-1] + 0.1
            delta_qbit_limit = qbit_size_limit - qbit_size_limits[-1] + 0.1
        else:
            delta_actual_size = 1
            delta_qbit_limit = 1
            
        actual_sizes.append(actual_size)
        qbit_size_limits.append(qbit_size_limit)
        searched_qbit_size = final_qbit_size_estimator(final_qbits.detach().cpu().numpy(), split.detach().cpu().numpy(), gaussians.n_channel, n_block, enable_drop)
        logger.info(f"{j} searched_qbit_size: {searched_qbit_size / 8 } bytes")
        delta_qbit = delta_qbit_limit / delta_actual_size * (model_size_limit - actual_size)
        if delta_qbit > actual_size:
            delta_qbit = model_size_limit - actual_size
        qbit_size_limit = qbit_size_limit + delta_qbit
        
        logger.info(f"{j} actual size: {actual_size / 8 / 1024/1024} {actual_size / 8} B")
        if abs(actual_size - model_size_limit) / model_size_limit < 0.05:
            return final_qbits
    return final_qbits


def compute_entropy_np(data):
    """
    Faster entropy computation using numpy.
    """
    values, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))  # 避免 log(0)
    return entropy
         
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []
    psnrs = []
    ssims = []
    lpipss = []
    # psnrs.append(psnr(renders[idx], gts[idx]))
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)

        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        psnrs.append(psnr(rendering, gt))
        ssims.append(ssim(rendering, gt))
        lpipss.append(lpips_fn(rendering, gt))
        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    ave_psnr = torch.stack(psnrs).mean()
    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')
    print(f'PSNR {ave_psnr}')

    ssim_val = torch.tensor(ssims).mean()
    psnr_val = torch.stack(psnrs).mean()
    lpips_val = torch.tensor(lpipss).mean()
        
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      
    print(psnr_val, ssim_val, lpips_val)
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, logger, input_path : str = None, name : str = None):
	dataset.mesongs = False
	load_compressed=True
	with torch.no_grad():
		gaussians = GaussianModel(
			scene_name=dataset.scene_name, 
			feat_dim=dataset.feat_dim, 
			n_offsets=dataset.n_offsets, 
			voxel_size=dataset.voxel_size, 
			update_depth=dataset.update_depth, 
			update_init_factor=dataset.update_init_factor, 
			update_hierachy_factor=dataset.update_hierachy_factor, 
			use_feat_bank=dataset.use_feat_bank, 
			appearance_dim=dataset.appearance_dim, 
			ratio=dataset.ratio, 
			add_opacity_dist=dataset.add_opacity_dist, 
			add_cov_dist=dataset.add_cov_dist, 
			add_color_dist=dataset.add_color_dist,
			enable_drop=dataset.enable_drop,
			use_pcc=True, 
			logger=logger)
		scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_compressed=load_compressed)
		
		gaussians.eval()
  
		C = torch.concat([
            gaussians._anchor_feat.detach(), #32
            gaussians._offset.detach().flatten(-2).contiguous(), # 30
            gaussians._opacity.detach().contiguous(), #1
            gaussians._scaling.detach(), #6
            gaussians._rotation.detach()], -1) #4
		channel_entropies = []
		for i in range(C.shape[0]):
			channel_entropies.append(compute_entropy(C[i]))

		

		bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
		background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
		if not os.path.exists(dataset.model_path):
			os.makedirs(dataset.model_path)

		render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
		zip_path = os.path.join(dataset.model_path, 'zip', f'iteration_{iteration}', 'bins.zip')
		sz = os.path.getsize(zip_path)
		print(f'size: {sz / 1024 / 1024} MB')
    # A = torch.tensor(np.load("ingredients/A.npy"), dtype=int, device="cuda")
    # final_qbits = torch.tensor(np.load("ingredients/final_qbits.npy"), dtype=int, device="cuda")
    # model_size_limit = (dataset.target_size - 2) * 1024 * 1024 * 8
    # splits = split_length(gaussians.n_rf, gaussians.n_block)
    # splits = torch.tensor([splits]*gaussians.n_channel, device='cuda', dtype=int)
    # cha_qbits = research_fine(
    #     A,
    #     final_qbits,
    #     gaussians,
    #     scene, 
    #     model_size_limit,
    #     gaussians.bit_lower_bound,
    #     gaussians.n_bit,
    #     n_block=gaussians.n_block,
    #     enable_drop=True,
    #     split=splits,
    # )
    # gaussians.update_bqa_wo_refresh(cha_qbits)
    # # scene.save_compressed(0)
    # zip_size, zip_path = scene.save_compressed(0, return_path=True)
    # gaussians.load_compressed(zip_path)
    
    # render_set(dataset.model_path, "test_adapt", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
    
    # if (not skip_train) and (input_path != "") and (input_path is not None):
    #     cam_traj = generate_path(scene.getTrainCameras(), n_frames=480)
    #     render_set(dataset.model_path, "train", scene.loaded_iter, cam_traj, gaussians, pipeline, background)
    #     base_dir = 'videos_temp'
    #     input_path = os.path.join(input_path, "ours_{}".format(scene.loaded_iter))
    #     print(input_path)
    #     name = f"{name}_{scene.loaded_iter}"
    #     create_videos(base_dir, input_path, name) 

        
def get_logger(path, iteration):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, f"render_{iteration}.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)
    return logger
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_iter", default=30_000, type=int)
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--name", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)
    logger = get_logger(model_path, args.iteration)
    # print(logger)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,logger,  args.input_path, args.name)
