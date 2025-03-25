import numpy as np 
import torch
from pulp import LpProblem, LpVariable, LpMinimize, LpInteger, LpStatus, PULP_CBC_CMD
from utils.quant_utils import split_length
from multiprocessing import Pool
import BDD.ILP_instance_py as ilp_instance
from BDD.bdd_solver_py import bdd_solver as bdd_solver
from BDD.bdd_solver_py import bdd_solver_options as bdd_solver_options

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
            qbsz += si * qbits[i*n_bit + j] * (j + low_bit)
    
    return qbsz # bit

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

    qbits = np.zeros(n_channel).astype(int)
    for i in range(n_channel):
        for j in range(n_bit):
            if variable[f"x{i*n_bit+j}"].varValue != 0:
                qbits[i] = j+low_bit 
    
    return qbits


def solve_coarse_ilp_bdd(L, n_bit, low_bit, n_channel, n_point, qbit_size_limit):
    n_variable = n_bit * n_channel
    objective = {}
    size_coni = {}
    assert len(L) == n_variable, "false L length"
    for i in range(n_variable):
        objective[f"x{i}"] = L[i]
        bit_width = (i % n_bit) + low_bit
        coni_arg = bit_width * (n_point - 1)
        size_coni[f"x{i}"] = coni_arg
        
    constraints = []
    constraints.append([
        'C1',
        size_coni,
        '<',
        qbit_size_limit
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
    opts.dual_time_limit = 200
    opts.incremental_primal_num_rounds = 4
    # initialize solver
    solver = bdd_solver(opts)
    # solve dual problem
    solver.solve_dual()
    lower_bound = solver.lower_bound()
    # run primal heuristic
    upper_bound, sol = solver.round()
    # extracting the solution as a numpy array
    sol_numpy = np.array(sol)
    print('sol_numpy', sol_numpy)
    sol_tensor = torch.tensor(sol_numpy).reshape(n_channel, n_bit)
    # print(sol_numpy)
    sol_tensor_sum = torch.sum(sol_tensor, dim=-1)
    # print('diff 1', torch.sum(torch.square(sol_tensor_sum - torch.ones_like(sol_tensor_sum))))

    qbits = np.zeros(n_channel).astype(int)
    for i in range(n_channel):
        for k in range(n_bit):
            if sol_numpy[i*n_bit + k] != 0:
                qbits[i] = k+low_bit
    
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

A = np.load('/home/szxie/SizeGS/exp_data/bicycle_A.npy') # [n_bit, C, B]

torchA = torch.tensor(A, device='cuda') 
B = torch.sum(torchA, dim=-1).transpose(0, 1) # [C, n_bit]
print(B)
B_flat = B.flatten().cpu().tolist()
torchA = torchA.permute(1, 2, 0)

A = torchA.cpu().numpy()

print(A.shape, torchA.shape) # must be [C, B, n_bit]

features = torch.tensor(np.load('exp_data/bicycle_after_raht.npy'), device='cuda')

n_channel, n_block, n_bit = torchA.shape
bit_lower_bound = 1
model_size_limit = 20*1024*1024*8
n_point = features.shape[0]
print('n_point', n_point)
splits = split_length(n_point-1, n_block)
splits = [splits] * n_channel
splits = torch.tensor(splits, device="cuda")
qbit_size_limit, remain_size = qbit_upper_size(n_point, n_channel, n_block, model_size_limit)

cha_qbits = solve_coarase_ilp_pulp(
    B_flat,
    n_bit = n_bit,
    low_bit = bit_lower_bound,
    n_channel = n_channel,
    n_block = n_block,
    n_point = n_point,
    qbit_size_limit = qbit_size_limit
)


cha_qbits_bdd = solve_coarse_ilp_bdd(
    B_flat,
    n_bit, 
    bit_lower_bound,
    n_channel,
    n_point,
    qbit_size_limit
)


print(cha_qbits_bdd)
print(cha_qbits)

np.save('exp_data/cha_qbits_bdd.npy', cha_qbits_bdd)
np.save('exp_data/cha_qbits_pulp.npy', cha_qbits)

# # print(cha_qbits)
per_channel_size_limit = per_channel_qbit_upper(cha_qbits, qbit_size_limit)

# 将每个通道的处理封装为函数
def solve_wrapper(args):
    L, n_bit, low_bit, n_block, n_point, splits, model_size_limit = args
    return solve_channel_ilp_pulp(
        L=L,
        n_bit=n_bit,
        low_bit=low_bit,
        n_block=n_block,
        n_point=n_point,
        splits=splits,
        model_size_limit=model_size_limit
    )

# 准备参数列表（在父进程中进行数据预处理）
args_list = [
    (
        torchA[i].flatten().cpu().tolist(),  # 提前转换为CPU列表
        n_bit,
        bit_lower_bound,
        n_block,
        n_point,
        splits[i].reshape(-1).cpu().tolist(),
        pc_szlmit
    )
    for i, pc_szlmit in enumerate(per_channel_size_limit)
]

# 创建进程池并行执行
with Pool() as pool:  # 自动使用所有CPU核心
    results = pool.map(solve_wrapper, args_list)

# 在父进程中将结果转换为CUDA张量
final_qbits = [torch.tensor(res, device="cuda", dtype=int) for res in results]

# final_qbits = torch.stack(final_qbits)
print(final_qbits)


# arxiv
# final_qbits = []
# for i, pc_szlmit in enumerate(per_channel_size_limit):
#     per_block_qbits = solve_channel_ilp_pulp(
#         L=A[i].flatten().cpu().tolist(),
#         n_bit=n_bit,
#         low_bit=bit_lower_bound,
#         n_block=n_block,
#         n_point=n_point,
#         splits=splits[i].reshape(-1).cpu().tolist(),
#         model_size_limit=pc_szlmit
#     )
#     final_qbits.append(torch.tensor(per_block_qbits, device="cuda", dtype=int))
