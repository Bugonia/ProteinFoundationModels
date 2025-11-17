import os
import sys

# 导入您包含所有类的代码文件 (假设它叫 esm_data.py)
try:
    from data import ESMStructuralSplitDataset
except ImportError:
    print("错误：找不到 esm_data.py。请确保您的类定义保存在 'esm_data.py' 文件中。")
    sys.exit(1)


# -----------------------------------------------------------------
# 这是关键：手动定义您的共享缓存路径
# -----------------------------------------------------------------
# 1. 获取您的个人全局目录
SHARED_GLOBAL_DIR = "/inspire/hdd/global_user/zhongxiaoqiu-253108120179"

# 2. 定义 torch 应该使用的缓存根目录
#    我们将遵循 'Tip 3' 的建议
TORCH_CACHE_ROOT = os.path.join(SHARED_GLOBAL_DIR, ".cache/torch")

# 3. 定义这个特定数据集的路径
DATASET_PATH = os.path.join(TORCH_CACHE_ROOT, "data/esm")
# -----------------------------------------------------------------


print(f"开始下载数据集...")
print(f"将要保存到 (共享路径): {DATASET_PATH}")

# 确保父目录存在
os.makedirs(DATASET_PATH, exist_ok=True)

try:
    dataset = ESMStructuralSplitDataset(
        split_level="family",
        cv_partition="1",
        split="train",
        root_path=DATASET_PATH,  # <--- 在这里传入我们自定义的路径
        download=True            # <--- 确保下载
    )
    
    print("\n--- 下载完成！ ---")
    print(f"数据已保存到: {dataset.base_path}")
    print("\n测试读取一个样本:")
    print(dataset[0]['seq'][:50] + "...") # 加载第一个样本进行测试

except Exception as e:
    print(f"\n下载或解压过程中出错: {e}")