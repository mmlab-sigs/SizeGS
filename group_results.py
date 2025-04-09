import json
import os
import pandas as pd

def find_max_psnr_entry(json_data):
    """从JSON数据中找到PSNR最大的条目"""
    return max(json_data.items(), key=lambda x: x[1]['PSNR'])

def process_json_files(json_file_paths, log_names, scene_names, output_file):
    """处理目录中的所有JSON文件并生成Excel"""
    data_list = []
    
    i = 0
    # 遍历目录中的所有JSON文件
    for file_path, log_name in zip(json_file_paths, log_names):
        with open(file_path, 'r') as f:
            try:
                json_data = json.load(f)
                max_entry = find_max_psnr_entry(json_data)
                
                # 构建数据行
                row = {
                    '文件名': log_name,
                    '条目名称': max_entry[0],
                    'SCENE': scene_names[i],
                    'SSIM': max_entry[1]['SSIM'],
                    'PSNR': max_entry[1]['PSNR'],
                    'LPIPS': max_entry[1]['LPIPS'],
                    'SIZE': max_entry[1]['SIZE'],
                    'SIZE (bytes)': max_entry[1]['SIZE (bytes)']
                }
                data_list.append(row)
                
            except Exception as e:
                print(f"处理文件 {log_name} 时出错: {str(e)}")
        i+=1
    
    # 创建DataFrame并保存Excel
    if data_list:
        df = pd.DataFrame(data_list)
        # 按PSNR降序排序
        # df = df.sort_values(by='PSNR', ascending=False)
        df.to_excel(output_file, index=False)
        print(f"结果已保存至 {output_file}")
    else:
        print("未找到有效的JSON文件")

if __name__ == "__main__":
    # 配置参数
    json_directory = 'json_files'  # JSON文件所在目录
    output_excel = 'exp_data/max_psnr_results.xlsx'  # 输出文件名
    scene_names = ["bicycle","bonsai","counter","flowers","garden","kitchen","room","stump","treehill"]
    # log_names = [
    #     '2025-03-28_18:29:51',
    #     '2025-03-29_15:13:00',
    #     '2025-03-29_15:55:39',
    #     '2025-03-29_19:07:19',
    #     '2025-03-29_16:34:21',
    #     '2025-03-29_17:13:32',
    #     '2025-03-29_17:55:16',
    #     '2025-03-29_18:34:26',
    #     '2025-03-29_19:38:20'
    # ]
    log_names = [
        '2025-03-29_22:43:10',
        '2025-03-29_23:32:16',
        '2025-03-30_00:22:01',
        '2025-03-30_04:05:12',
        '2025-03-30_01:06:37',
        '2025-03-30_01:52:22',
        '2025-03-30_02:42:07',
        '2025-03-30_03:29:29',
        '2025-03-30_04:41:50'
    ]
    json_file_paths = []
    prefix = 'outputs/360_v2'
    for i, sc in enumerate(scene_names):
        json_file_paths.append(os.path.join(prefix, sc, 'sizegs_gpcc_no_raht', log_names[i], 'results.json'))
    
    # 执行处理
    process_json_files(json_file_paths, log_names, scene_names, output_excel)