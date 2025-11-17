import os
import sys
import torch

# 1. 导入您的数据集类
try:
    # 确保 data.py 和这个脚本在同一个目录
    from data import ESMStructuralSplitDataset 
except ImportError:
    print("❌ 错误: 找不到 'data.py'。")
    print("请确保包含 ESMStructuralSplitDataset 类的文件 'data.py' 与此脚本在同一目录中。")
    # ...

print("--- 1. 硬件测试 (GPU) ---")

# 2. 检查 PyTorch 和 CUDA
print(f"PyTorch 版本: {torch.__version__}")
is_cuda = torch.cuda.is_available()
print(f"CUDA 是否可用: {is_cuda}")

if not is_cuda:
    print("❌ 警告: PyTorch 无法在此机器上找到 CUDA。")
    print("   您的代码可以运行，但将使用 CPU。")
else:
    print(f"✅ 成功: 找到 GPU: {torch.cuda.get_device_name(0)}")

print("\n--- 2. 文件系统测试 (数据) ---")

# 3. 严格定义您的共享数据路径 (!!关键!!)
#    我们不能使用 '~' 因为您是 root 用户
SHARED_DATA_PATH = "/inspire/hdd/global_user/zhongxiaoqiu-253108120179/.cache/torch/data/esm"

print(f"正在尝试访问共享路径: {SHARED_DATA_PATH}")

try:
    # 4. 尝试加载数据集
    dataset = ESMStructuralSplitDataset(
        split_level="family",       # 使用与下载时相同的设置
        cv_partition="1",
        split="train",
        root_path=SHARED_DATA_PATH, # 传入我们的自定义共享路径
        download=False              # (!!极其重要!!) 设置为 False，因为此机器无法上网
    )
    
    print(f"✅ 成功: 数据集 'splits' 文件加载成功。")
    print(f"   找到 {len(dataset)} 个训练样本。")
    
    # 5. 尝试从 .pkl 文件中读取第一个样本
    print("   正在尝试读取第一个样本 (.pkl 文件)...")
    sample = dataset[0]
    
    print(f"✅ 成功: 样本数据读取成功。")
    print(f"   样本序列 (前50字符): {sample['seq'][:50]}...")

except FileNotFoundError:
    print(f"❌ 失败: 在 {SHARED_DATA_PATH} 找不到数据。")
    print("   请确认您已在 CPU 实例上成功运行了 download_data.py，")
    print("   并且它下载到了 *完全相同* 的共享路径。")
except Exception as e:
    print(f"❌ 失败: 加载数据集时发生意外错误: {e}")

print("\n--- 测试完成 ---")