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
import numpy as np

import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
import copy
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, LpStatus, PULP_CBC_CMD
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui, ft_render, generate_neural_gaussians, rendersave, renderload
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.tensorboard import SummaryWriter
import BDD.ILP_instance_py as ilp_instance
from BDD.bdd_solver_py import bdd_solver as bdd_solver
from BDD.bdd_solver_py import bdd_solver_options as bdd_solver_options
import numpy as np


# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')

def prune_mask(percent, imp):
    sorted_tensor, _ = torch.sort(imp, dim=0)
    index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
    value_nth_percentile = sorted_tensor[index_nth_percentile]
    prune_mask = (imp <= value_nth_percentile).squeeze()
    return prune_mask

def cal_imp(
        gaussians,
        views,
        pipe, 
        background
    ):
    
    anchor_size = gaussians.get_anchor.shape[0]
    n_offsets =  gaussians.n_offsets
    full_opa_imp = torch.zeros([anchor_size, n_offsets])
    full_opa_imp = full_opa_imp.contiguous().flatten()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipe,background)
        render_results = render(view, gaussians, pipe, background, visible_mask=voxel_visible_mask, meson_count=True)
        # render_results = render(view, gaussians, pipe, background, debug=False, clamp_color=True, meson_count=True, f_count=False, depth_count=False)
        anchor_filter = voxel_visible_mask
        offset_filter = render_results["selection_mask"]
        anchor_filter = anchor_filter.view(-1, 1).repeat([1, n_offsets]).reshape(-1)
        anchor_filter = anchor_filter.to('cpu')
        offset_filter = offset_filter.to('cpu')
        intermediate_imp = copy.deepcopy(full_opa_imp[anchor_filter])
        intermediate_imp[offset_filter] += render_results["imp"].detach().cpu()
        full_opa_imp[anchor_filter] = intermediate_imp
        
        del intermediate_imp
        del render_results
    
    full_opa_imp = full_opa_imp.reshape([gaussians.get_anchor.shape[0], gaussians.n_offsets]).sum(-1, keepdim=False)
    
    return full_opa_imp.detach()


def Size(n, qbits, n_block, offset):
    qbit = 0
    for i in range(73):
        for j in range(16):
            qbit += qbits[i*16+j]*(j+1)
    return ((n-1)*qbit + 73*32 + n*64 + 73*n_block*32*2 + 2*32 + n_block*32)/8 + 50*1024 + 4*n_block +offset*1024*1024

def finalSize(n, qbits, n_block, offset):
    qbit = sum(qbits)
    return ((n-1)*qbit + 73*32 + n*64 + 73*n_block*32*2 + 2*32 + n_block*32)/8 + 50*1024 + 4*n_block +offset*1024*1024

def get_percent(target_size, qbit, n_block, anchor_size, offset):
    n = int((target_size - 50*1024 - 4*n_block - offset*1024*1024)*8 - 73*32 - 73*n_block*32*2 - 2*32 - n_block*32 + qbit)/(qbit+64)
    return (anchor_size - n)/anchor_size

def newSize(n, n_block, target_size, offset=0):
    remain = (73*32 + n*64 + 73*n_block*32*2 + 2*32 )/8 + 50*1024 + offset*1024*1024
    # Todo: rewrite the new Size function.
    # return qbits < ?
    return (target_size - remain) * 8 # bit
    
def qbit_upper_size(n_point, n_channel, n_block, target_size):
    trans_size = n_channel * n_block * 2 * 32 + 2 * 32 # scale and zero points, n_block, depth
    channel_check = n_channel * n_block * 8
    oct_param = 6 * 32
    dc_size = n_channel * 32 # DC coefficient
    oct_size = n_point * 64 # octree size 
    mlp_size = 50*1024
    remain_size = mlp_size * 8 + trans_size + channel_check + oct_param + dc_size + oct_size

    return target_size - remain_size, remain_size # bit

def per_channel_qbit_upper(cha_qbits, qbit_size_limit):
    qbitsum = sum(cha_qbits)
    elem = qbit_size_limit / qbitsum 
    per_channel_uppers = []
    for bi in cha_qbits:
        per_channel_uppers.append(bi * elem)
    return per_channel_uppers

def qbit_channel_size_estimator(qbits, n_point, n_bit, low_bit, n_channel, remain_size):
    qbsum = 0
    for i in range(n_channel):
        for j in range(n_bit):
            qbsum += qbits[i*n_bit + j] * (j + low_bit)
    
    return remain_size + qbsum * (n_point - 1) # bit

def qbit_block_size_esimator(qbits, n_bit, low_bit, splits):
    # qbits: [0 or 1]
    qbsz = 0
    for i, si in enumerate(splits):
        for j in range(n_bit):
            qbsz += si * qbits[i*16 + j] * (j + low_bit)
    
    return qbsz # bit

def solve_finer_01_ilp_problem(L, n_bit, low_bit, n_channel, n_block, splits, qbit_size):
    n_variable = n_bit * n_channel * n_block
    objective = {}
    size_coni = {}
    for i in range(n_variable):
        objective[f"x{i}"] = L[i]
        re = (i % n_bit) + 1
        cbi = int(i // n_bit)
        ci = int(cbi // n_block) 
        bi = int(cbi % n_block)
        coni_arg = re * int(splits[ci, bi].item()) / 8
        size_coni[f"x{i}"] = coni_arg
        
    constraints = []
    # TODO: add size constraint
    # n_cons = 1
    constraints.append([
        f'C1',
        size_coni,
        '<',
        qbit_size
    ])
    for i in range(n_channel * n_block):
        coni = [f'C{i+2}']
        coni_dict = {}
        for j in range(n_bit):
            coni_dict[f"x{i*n_bit+j}"] = 1.0
        coni.append(coni_dict)
        coni.append('=')
        coni.append(1.0)
        # print(coni)
        constraints.append(coni)
    
    # objective = {'x1': 2.0, 'x2': 1.0, 'x3': -1.0, 'x4': 1.0, 'x5': -2.0, 'x6': -5.0}
    # constraints = [
    #     ['C1', {'x1': 1.0, 'x2': 1.0, 'x3': 1.0}, '=', 1.0],
    #     ['C2', {'x3': 1.0, 'x4': 1.0, 'x5': 1.0}, '<', 1.0],
    #     ['C3', {'x4': 1.0, 'x5': 1.0, 'x6': 1.0}, '>', 1.0]
    # ]
    ilp = ilp_instance.ILP_instance()
    for var_name, coeff in objective.items():
        ilp.add_new_variable_with_obj(var_name, coeff)

    for con in constraints:
        con_name = con[0]
        con_vars = list(con[1].keys())
        con_coeffs = list(con[1].values())
        con_coeffs = [int(c) for c in con_coeffs] # Only integer coefficients are supported.
        con_sense = con[2]
        rhs_value = int(con[3]) # Only integer right hand side value is supported.
        if con_sense == '=':
            ineq_type = ilp_instance.equal
        elif con_sense == '<':
            ineq_type = ilp_instance.smaller_equal
        else:
            assert(con_sense == '>')
            ineq_type = ilp_instance.greater_equal
        ilp.add_new_constraint(con_name, con_vars, con_coeffs, rhs_value, ineq_type)
    
    # passing ilp instance to solver options
    opts = bdd_solver_options(ilp)
    # this setting enables finding a primal feasible solution. If this is false only a lower bound will be computed.
    opts.incremental_primal_rounding = True
    # initialize solver
    solver = bdd_solver(opts)
    # solve dual problem
    solver.solve_dual()
    lower_bound = solver.lower_bound()
    # run primal heuristic
    upper_bound, sol = solver.round()
    # extracting the solution as a numpy array
    sol_numpy = np.array(sol)
    
    sol_tensor = torch.tensor(sol).reshape(n_channel, n_block, n_bit)
    # print(sol_numpy)
    sol_tensor_sum = torch.sum(sol_tensor, dim=-1)
    # print('diff 1', torch.sum(torch.square(sol_tensor_sum - torch.ones_like(sol_tensor_sum))))

    qbits = np.zeros([n_channel, n_block]).astype(int)
    for i in range(n_channel):
        for j in range(n_block):
            for k in range(n_bit):
                if sol_numpy[i*n_block*n_bit + j*n_bit + k] != 0:
                    qbits[i, j] = k+low_bit
                    
    return qbits

def solve_coarse_01_ilp_problem(L, n_bit, low_bit, n_channel, n_point, qbit_size):
    n_variable = n_bit * n_channel
    objective = {}
    size_coni = {}
    assert len(L) == n_variable, "false L length"
    for i in range(n_variable):
        objective[f"x{i}"] = L[i]
        re = (i % n_bit) + 1
        coni_arg = re * (n_point - 1)
        size_coni[f"x{i}"] = coni_arg
        
    constraints = []
    constraints.append([
        f'C1',
        size_coni,
        '<',
        qbit_size
    ])
    # TODO: add size constraint
    # n_cons = 1
    for i in range(n_channel):
        coni = [f'C{i+2}']
        coni_dict = {}
        for j in range(n_bit):
            coni_dict[f"x{i*n_bit+j}"] = 1
        coni.append(coni_dict)
        coni.append('=')
        coni.append(1)
        # print(coni)
        constraints.append(coni)
    
    ilp = ilp_instance.ILP_instance()
    for var_name, coeff in objective.items():
        ilp.add_new_variable_with_obj(var_name, coeff)

    for con in constraints:
        con_name = con[0]
        con_vars = list(con[1].keys())
        con_coeffs = list(con[1].values())
        con_coeffs = [int(c) for c in con_coeffs] # Only integer coefficients are supported.
        con_sense = con[2]
        rhs_value = int(con[3]) # Only integer right hand side value is supported.
        if con_sense == '=':
            ineq_type = ilp_instance.equal
        elif con_sense == '<':
            ineq_type = ilp_instance.smaller_equal
        else:
            assert(con_sense == '>')
            ineq_type = ilp_instance.greater_equal
        ilp.add_new_constraint(con_name, con_vars, con_coeffs, rhs_value, ineq_type)
    
    # passing ilp instance to solver options
    opts = bdd_solver_options(ilp)
    # this setting enables finding a primal feasible solution. If this is false only a lower bound will be computed.
    opts.incremental_primal_rounding = True
    # initialize solver
    solver = bdd_solver(opts)
    # solve dual problem
    solver.solve_dual()
    lower_bound = solver.lower_bound()
    # run primal heuristic
    upper_bound, sol = solver.round()
    # extracting the solution as a numpy array
    sol_numpy = np.array(sol)
    
    sol_tensor = torch.tensor(sol).reshape(n_channel, n_bit)
    # print(sol_numpy)
    sol_tensor_sum = torch.sum(sol_tensor, dim=-1)
    # print('diff 1', torch.sum(torch.square(sol_tensor_sum - torch.ones_like(sol_tensor_sum))))

    qbits = np.zeros(n_channel).astype(int)
    for i in range(n_channel):
        for k in range(n_bit):
            if sol_numpy[i*n_bit + k] != 0:
                qbits[i] = k+low_bit
    
    return qbits

def solve_coarase_ilp_pulp(L, n_bit, low_bit, n_channel, n_block, n_point, qbit_size_limit):
    num_variable = n_channel * n_bit 
    variable = {}
    for i in range(num_variable):
        variable[f"x{i}"] = LpVariable(f"x{i}", 0, 1, cat=LpInteger)
    prob = LpProblem("Model_Size", LpMinimize)

    prob += qbit_channel_size_estimator([variable[f"x{i}"] for i in range(num_variable)], n_point, n_bit, low_bit, n_channel, 0) <= qbit_size_limit

    for le in range(0, n_channel):
        ri = le * n_bit + n_bit
        prob += sum([variable[f"x{i}"] for i in range(le * n_bit, ri)]) == 1

    prob += sum([(variable[f"x{i}"]) * L[i] for i in range(num_variable)])

    time_limit = 200

    # 使用CBC求解器并设置时间限制
    prob.solve(PULP_CBC_CMD(timeLimit=time_limit, msg=True))

    solution = {}
    for i in range(num_variable):
        solution[f"x{i}"] = variable[f"x{i}"].varValue

    qbits = [0]*n_channel
    for i in range(n_channel):
        for j in range(n_bit):
            if variable[f"x{i*n_bit+j}"].varValue != 0:
                qbits[i] = j+low_bit 
    
    return qbits

def solve_channel_ilp_pulp(L, n_bit, low_bit, n_block, n_point, splits, model_size_limit):
    num_variable = n_block * n_bit 
    variable = {}
    for i in range(num_variable):
        variable[f"x{i}"] = LpVariable(f"x{i}", 0, 1, cat=LpInteger)
    prob = LpProblem("Model_Size", LpMinimize)

    prob += qbit_block_size_esimator([variable[f"x{i}"] for i in range(num_variable)], n_bit, low_bit, splits) <= model_size_limit

    for le in range(0, n_block):
        ri = le * n_bit + n_bit
        prob += sum([variable[f"x{i}"] for i in range(le * n_bit, ri)]) == 1

    prob += sum([(variable[f"x{i}"]) * L[i] for i in range(num_variable)])

    time_limit = 200

    # 使用CBC求解器并设置时间限制
    prob.solve(PULP_CBC_CMD(timeLimit=time_limit, msg=True))

    solution = {}
    for i in range(num_variable):
        solution[f"x{i}"] = variable[f"x{i}"].varValue

    qbits = [0]*n_block
    for i in range(n_block):
        for j in range(n_bit):
            if variable[f"x{i*n_bit+j}"].varValue != 0:
                qbits[i] = j+low_bit 
    
    return qbits

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, load_iteration, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # final_qbits = [0]*73
    imp = None
    offset = 0 #debug only

    bg = time.time()
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, 
        dataset.voxel_size, dataset.update_depth, 
        dataset.update_init_factor, dataset.update_hierachy_factor, 
        dataset.use_feat_bank, dataset.appearance_dim, 
        dataset.ratio, dataset.add_opacity_dist, 
        dataset.add_cov_dist, dataset.add_color_dist, 
        dataset.raht, pipe.n_block, dataset.unit_length, 
        dataset.use_pcc, logger)
    
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    all_anchor_size = int(gaussians._anchor.shape[0])
    
    logger.info('load gaussian model time: {}'.format(time.time() - bg))
    
    bg = time.time()
    with torch.no_grad():
        if imp is None:
            gaussians.eval()
            imp = cal_imp(gaussians, scene.getTrainCameras(), pipe, background)
            # print(imp.shape)
        imp = imp.detach()
        # purne_percent = get_percent(dataset.target_size*1024*1024, best_qbit*73, pipe.n_block, all_anchor_size, offset)
        purne_percent = 0.6
        # print(purne_percent)
        pmask = prune_mask(purne_percent, imp)
        remain_imp = imp[torch.logical_not(pmask)]
        gaussians.prune_anchor_no_grad(pmask)
    logger.info('cal imp: {}'.format(time.time() - bg))
    
    
    # start after-pruning warm up
    gaussians.training_setup(opt)
    gaussians.train()
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    writer = SummaryWriter(log_dir='runs/testing')
    offset_iteration = 0
    viewpoint_stack = None
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    
    ema_loss_for_log = 0.0
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.prune_iterations), desc="prune warm progress")
    first_iter += 1
    # dp=False
    logger.info("start prune warm up training")
    
    # warm up
    for iteration in range(first_iter, opt.prune_iterations+1):
        iter_start.record()
        gaussians.optimizer.zero_grad()
        gaussians.update_learning_rate(iteration)
        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background, training=True)
                
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            visible_mask=voxel_visible_mask, 
            retain_grad=False
        )
        image, scaling = render_pkg["render"], render_pkg["scaling"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
        loss.backward()
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.prune_iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
                    
            # Optimizer step
            if iteration < opt.prune_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

        writer.add_scalar('training loss', loss.item(), iteration + offset_iteration)
    offset_iteration += opt.prune_iterations
    
    # disable anchor update 
    if not pipe.enable_octree_training:
        gaussians.octree_offline(remain_imp.clone(), dataset.oct_merge)

    # start after-octree warm up
    gaussians.training_setup(opt)
    gaussians.train()
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    ema_loss_for_log = 0.0
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.octree_iterations), desc="Octree warm up progress")
    first_iter += 1
    
    # warm up
    for iteration in range(1, opt.octree_iterations+1):
        gaussians.optimizer.zero_grad()
        gaussians.update_learning_rate(iteration)
        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background, training=True)
                
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            visible_mask=voxel_visible_mask, 
            retain_grad=False
        )
        image, scaling = render_pkg["render"], render_pkg["scaling"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
        loss.backward()
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.octree_iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
                    
            # Optimizer step
            if iteration < opt.octree_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        writer.add_scalar('training loss', loss.item(), iteration + offset_iteration)
    
    offset_iteration += opt.octree_iterations
    
    model_size_limit = dataset.target_size*1024*1024*8
    num_variable = gaussians.n_channel * gaussians.n_block * gaussians.n_bit

    A, B = gaussians.get_fine_grained_ddrf()
    td = True
    if td:
        br = time.time()
        
        # print(A.shape)
        # print(len(B))
        # assert len(A) == num_variable, "Length of L does not match number of variables."
        
        # qbit_upper_size = newSize(gaussians.n_point, gaussians.n_block, model_size_limit, offset=offset)
        # print('qbit_upper_size', qbit_upper_size)
        # bit
        qbit_size_limit, remain_size = qbit_upper_size(gaussians.n_point, gaussians.n_channel, gaussians.n_block, model_size_limit)
        
        cha_qbits = solve_coarase_ilp_pulp(
            B,
            n_bit = gaussians.n_bit,
            low_bit = gaussians.bit_lower_bound,
            n_channel = gaussians.n_channel,
            n_block = gaussians.n_block,
            n_point = gaussians.n_point,
            qbit_size_limit = qbit_size_limit
        )
        
        # print(cha_qbits)
        per_channel_size_limit = per_channel_qbit_upper(cha_qbits, qbit_size_limit)
        
        final_qbits = []
        for i, pc_szlmit in enumerate(per_channel_size_limit):
            per_block_qbits = solve_channel_ilp_pulp(
                L=A[i].flatten().cpu().tolist(),
                n_bit=gaussians.n_bit,
                low_bit=gaussians.bit_lower_bound,
                n_block=gaussians.n_block,
                n_point=gaussians.n_point,
                splits=gaussians.dp_split[i].reshape(-1).cpu().tolist(),
                model_size_limit=pc_szlmit
            )
            final_qbits.append(torch.tensor(per_block_qbits, device="cuda", dtype=int))
        final_qbits = torch.stack(final_qbits)
        np.save('exp_data/final_qbits.npy', final_qbits.cpu().numpy())
        logger.info('searching time: {}'.format(time.time() - br))
    else:
        final_qbits = torch.tensor(np.load('exp_data/final_qbits.npy'), device="cuda", dtype=int)
    
    # np.save('exp_data/final_qbits.npy', final_qbits.cpu().numpy())
    # print(final_qbits)
    # print('cha_qbits', cha_qbits)
    # exit(0)
    # print('cha_qbits', cha_qbits)
    # print(qbits)
    # print('qbits.shape', qbits.shape)
    # final_qbits = cha_qbits
    
    #debug
    # final_qbits = [8]*73
    # print("final qbits", np.sum(final_qbits)/gaussians.n_channel/gaussians.n_block)
    # print("final qbits", final_qbits[:10, :10])

    gaussians.init_bqa(final_qbits)
    gaussians.training_setup(opt)
    gaussians.train()

    logger.info("\n[ITER 0] Saving Gaussians")
    with torch.no_grad():
        scene.save_compressed(0)

    octree_train_start = torch.cuda.Event(enable_timing = True)
    octree_train_end = torch.cuda.Event(enable_timing = True)
    change_dp_start = torch.cuda.Event(enable_timing = True)
    change_dp_end = torch.cuda.Event(enable_timing = True)
    prefilter_voxel_start = torch.cuda.Event(enable_timing = True)
    prefilter_voxel_end = torch.cuda.Event(enable_timing = True)
    loss_best = 0.05
    best_iteration = 0

    # writer = SummaryWriter(log_dir='runs/testing')
    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter = 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # dp=False
    logger.info("start Training")
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    for iteration in range(first_iter, opt.iterations + 1):
        
        iter_start.record()
        gaussians.optimizer.zero_grad()
        gaussians.update_learning_rate(iteration)
        
        # octree_train_start.record()
        # 
        # octree_train_end.record()
        
        # if iteration == 1:
            
            # gaussians.init_qas(pipe.n_block, final_qbits)
            # gaussians.update_bqa(final_qbits)
            # gaussians.set_dp_split(dp=dp)
            # print(gaussians.dp_split[0])

        # change_dp_start.record()
        # gaussians.change_dp_split(dp=dp)
        # change_dp_end.record()
        
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # prefilter_voxel_start.record()
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background, training=True)
        # prefilter_voxel_end.record()
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        # visible_mask = torch.ones(gaussians.get_anchor_v.shape[0], dtype=torch.bool, device = gaussians.get_anchor_v.device)
        
        render_pkg = ft_render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
        loss.backward()
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        iter_end.record()
        # logger.info("Training octree_train_start {}".format(octree_train_start.elapsed_time(octree_train_end)))
        # logger.info("Training change_dp_start {}".format(change_dp_start.elapsed_time(change_dp_end)))
        # logger.info("Training prefilter_voxel_start {}".format(prefilter_voxel_start.elapsed_time(prefilter_voxel_end)))
        # logger.info("Training iter {}".format(iter_start.elapsed_time(iter_end)))
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()


            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save_compressed(iteration)

            if loss < loss_best and iteration > opt.iterations-500:
                logger.info("\n[ITER {}] Saving Gaussians loss best".format(iteration))
                scene.save_compressed(iteration)
                loss_best = loss
                best_iteration = iteration
                logger.info("\n[ITER {}] loss best : {}".format(iteration, loss_best))

                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        writer.add_scalar('training loss', loss.item(), iteration + offset_iteration)
    writer.close()
    return best_iteration

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)
    if not os.path.exists(error_path):
        os.makedirs(error_path)

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []
    visible_count_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)

        rendering = render_pkg["render"]
        
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        gt = view.original_image[0:3, :, :]
        errormap = (rendering - gt).abs()
        name_list.append('{0:05d}'.format(idx) + ".png")
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)      
    
    return t_list, visible_count_list
    
# def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
#     makedirs(render_path, exist_ok=True)
#     makedirs(error_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)
    
#     t_list = []
#     visible_count_list = []
#     name_list = []
#     per_view_dict = {}
#     for idx, view in enumerate(tqdm(views, desc="Rendering save progress")):
        
#         torch.cuda.synchronize();t_start = time.time()
        
#         voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
#         torch.cuda.synchronize()
#         rendersave(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, idx=idx)
#         torch.cuda.synchronize();t_end = time.time()
#         t_list.append(t_end - t_start)

#     for idx, view in enumerate(tqdm(views, desc="Rendering load progress")):

#         # renders
#         torch.cuda.empty_cache()
#         render_pkg = renderload(idx = idx)
#         rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
#         visible_count = (render_pkg["radii"] > 0).sum()
#         visible_count_list.append(visible_count)


#         # gts
#         gt = view.original_image[0:3, :, :]
        
#         # error maps
#         errormap = (rendering - gt).abs()


#         name_list.append('{0:05d}'.format(idx) + ".png")
#         torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
#         torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
#         torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
#         per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
#         torch.cuda.empty_cache()

    
#     with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
#             json.dump(per_view_dict, fp, indent=True)
    
    # return t_list, visible_count_list



def render_sets(train_ds : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None, load_compressed=False):
    with torch.no_grad():
        dataset = copy.deepcopy(train_ds)
        dataset.mesongs = False
        dataset.raht = False
        
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, 
            dataset.voxel_size, dataset.update_depth, 
            dataset.update_init_factor, dataset.update_hierachy_factor, 
            dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.ratio, dataset.add_opacity_dist, 
            dataset.add_cov_dist, dataset.add_color_dist, 
            raht=False, n_block=pipeline.n_block, unit_length=dataset.unit_length, 
            use_pcc=dataset.use_pcc, logger=logger, enable_octree_training=False, 
            bit_upper_bound=dataset.bit_upper_bound, bit_lower_bound=dataset.bit_lower_bound)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_compressed=load_compressed)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{method_dir}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
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
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000,4000,5000,6000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10, 100, 1000, 5000, 6000, 7000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_iter", default=30_000, type=int)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    # if args.gpu != '-1':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    #     os.system("echo $CUDA_VISIBLE_DEVICES")
    #     logger.info(f'using GPU {args.gpu}')

        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # training
    best_iteration = training(lp.extract(args), op.extract(args), pp.extract(args), dataset, 
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, 
             args.load_iter, wandb, logger)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger, load_compressed=True)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")

    if best_iteration != 0:
        # rendering best
        logger.info(f'\nStarting Rendering best~')
        visible_count = render_sets(lp.extract(args), best_iteration, pp.extract(args), wandb=wandb, logger=logger, load_compressed=True)
        logger.info("\nRendering complete.")

        # calc metrics
        logger.info("\n Starting evaluation...")
        evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
        logger.info("\nEvaluating complete.")
