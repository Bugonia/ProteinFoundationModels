import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pyfaidx

# --- [!! 新 !!] 导入 DDP 和 Transformers 工具 ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup

# --- 1. 从您提供的文件中导入 ---
try:
    from .data import Alphabet
    from .esm2 import ESM2
except ImportError:
    print("错误: 确保 data.py 和 esm2.py 在同一目录中。")
    print("您是否从父目录使用了 'python -m ... '？")
    sys.exit(1)

# ----------------------------------------------------------------------
# 2. 高效的 Fasta 数据集 (与之前相同)
# ----------------------------------------------------------------------
class FastaDataset(Dataset):
    def __init__(self, fasta_file, is_main_process):
        # [!! 新 !!] 只让主进程打印，避免 8 个进程同时打印
        if is_main_process:
            print(f"正在打开 FASTA 文件: {fasta_file}")
        self.fasta = pyfaidx.Fasta(fasta_file)
        self.keys = list(self.fasta.keys())
        if is_main_process:
            print(f"成功加载 FASTA。找到 {len(self.keys)} 条序列。")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        seq_id = self.keys[idx]
        seq = str(self.fasta[seq_id])
        return seq

# ----------------------------------------------------------------------
# 3. MLM Data Collator (掩码器) (与之前相同)
# ----------------------------------------------------------------------
class MLMDataCollator:
    def __init__(self, alphabet: Alphabet, max_len: int, mlm_probability: float = 0.15):
        self.alphabet = alphabet
        self.max_len = max_len
        self.mlm_probability = mlm_probability
        self.pad_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.prob_mask = mlm_probability * 0.8
        self.prob_random = mlm_probability * 0.1

    def __call__(self, batch_seqs: list[str]):
        tokenized_seqs = [self.alphabet.encode(seq)[:self.max_len] for seq in batch_seqs]
        
        batch_tokens = []
        batch_labels = []

        for seq_tokens in tokenized_seqs:
            tokens = torch.tensor(seq_tokens, dtype=torch.long)
            labels = torch.full_like(tokens, -100)
            special_tok_mask = (
                (tokens == self.alphabet.cls_idx) |
                (tokens == self.alphabet.eos_idx) |
                (tokens == self.alphabet.padding_idx)
            )
            prob_matrix = torch.full(tokens.shape, self.mlm_probability)
            prob_matrix.masked_fill_(special_tok_mask, 0.0)
            masked_indices = torch.bernoulli(prob_matrix).bool()
            labels[masked_indices] = tokens[masked_indices]
            mask_mask = torch.bernoulli(torch.full(tokens.shape, 0.8)).bool() & masked_indices
            tokens[mask_mask] = self.mask_idx
            random_mask = torch.bernoulli(torch.full(tokens.shape, 0.5)).bool() & masked_indices & ~mask_mask
            random_tokens = torch.randint(
                low=self.alphabet.padding_idx + 1,
                high=self.alphabet.mask_idx - 1,
                size=tokens.shape,
                dtype=torch.long
            )
            tokens[random_mask] = random_tokens[random_mask]
            batch_tokens.append(tokens)
            batch_labels.append(labels)

        padded_tokens = torch.nn.utils.rnn.pad_sequence(
            batch_tokens, batch_first=True, padding_value=self.pad_idx
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=-100
        )
        
        current_max_len = padded_tokens.size(1)
        if current_max_len > self.max_len:
            padded_tokens = padded_tokens[:, :self.max_len]
            padded_labels = padded_labels[:, :self.max_len]
        
        return {"tokens": padded_tokens, "labels": padded_labels}

# ----------------------------------------------------------------------
# 4. 主训练脚本 (!! 已针对 DDP 更新 !!)
# ----------------------------------------------------------------------
def main():
    
    # --- A. [!! 新 !!] DDP 分布式设置 ---
    dist.init_process_group(backend="nccl") # 初始化 NCCL 后端 (用于 NVIDIA GPU)
    # DDP 会自动从环境变量中获取 'local_rank'
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # [!! 关键 !!] 只让主进程 (rank 0) 打印信息
    is_main_process = (local_rank == 0)
    
    if is_main_process:
        print("--- 1. 开始设置 [DDP 8-GPU 预训练] ---")
    
    # --- B. 路径和参数 ---
    FASTA_FILE_PATH = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/zhongxiaoqiu-253108120179/data/uniref50/uniref50.fasta"
    # [!! 新 !!] 为 DDP 训练创建一个全新的保存目录
    # 修改为新的目录，例如 epoch2
    SAVE_CHECKPOINT_PATH = "/inspire/hdd/global_user/zhongxiaoqiu-253108120179/models/esm2_150M_DDP_run_epoch2"
    
    # [!! 关键 !!] 充分利用您的 120 核 CPU
    # 12 个 "工人" * 8 块 GPU = 96 个 CPU 核心用于数据加载
    NUM_WORKERS = 12 
    
    MICRO_BATCH_SIZE = 16
    GRAD_ACCUMULATION_STEPS = 64 # 有效批量 = 16 * 64 = 1024
    MAX_LEN = 512
    EPOCHS = 1
    LEARNING_RATE = 5e-5 # (DDP 可以用稍大的学习率，但我们先保持 5e-5)
    
    if is_main_process:
        print(f"使用 {dist.get_world_size()} 块 GPU (H100)。")
        print(f"使用 {NUM_WORKERS * dist.get_world_size()} 个 CPU 核心用于数据加载。")
        os.makedirs(SAVE_CHECKPOINT_PATH, exist_ok=True)
    
    # --- C. 加载 Tokenizer (Alphabet) 和数据 ---
    if is_main_process:
        print("--- 2. 加载数据 ---")
    
    alphabet = Alphabet.from_architecture("ESM-1b")
    alphabet_size = len(alphabet)
    
    train_dataset = FastaDataset(FASTA_FILE_PATH, is_main_process)
    
    # [!! 新 !!] DDP 必须使用 DistributedSampler
    # sampler 会自动处理数据分片和 shuffle
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    data_collator = MLMDataCollator(alphabet, max_len=MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE,
        sampler=train_sampler, # <--- [!! 新 !!] 使用 sampler
        shuffle=False,         # <--- [!! 关键 !!] Sampler 负责 shuffle
        collate_fn=data_collator,
        num_workers=NUM_WORKERS, # <--- [!! 新 !!] 消除 I/O 瓶颈
        pin_memory=True
    )
    if is_main_process:
        print(f"数据加载器已准备就绪。共 {len(train_dataset)} 条序列。")
        print(f"每 GPU 有效批量大小: {MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS} (1024)")
        print(f"全局有效批量大小: {MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS * dist.get_world_size()} (8192)")
        print(f"最大序列长度: {MAX_LEN}")
    
    # --- D. 初始化模型、损失和优化器 ---
    if is_main_process:
        print("--- 3. 初始化模型 (从头开始) ---")
    
    model = ESM2(
        num_layers=30,
        embed_dim=640,
        attention_heads=8,
        alphabet=alphabet
    ).to(device)
    
    # ... (在 model = ESM2(...) 之后) ...

    # =================================================================
    # [新增] 加载 Epoch 1 的权重 (继续训练)
    # =================================================================
    # 指向您刚刚训练好的模型文件
    CHECKPOINT_PATH = "/inspire/hdd/global_user/zhongxiaoqiu-253108120179/models/esm2_150M_DDP_run/final_model.pt"
    
    # 检查文件是否存在
    if os.path.exists(CHECKPOINT_PATH):
        # 为了防止所有 GPU 同时打印，只让主进程 (Rank 0) 打印日志
        # (假设您的代码里有 rank 变量，如果没有，可以去掉 if rank == 0 判断)
        if 'rank' in locals() and rank == 0:
            print(f"--- 正在从 {CHECKPOINT_PATH} 加载 Epoch 1 权重 ---")
            
        # 1. 加载到 CPU
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        
        # 2. [关键] 处理 DDP 的 'module.' 前缀
        # 您保存的模型是因为 DDP 训练出来的，所以 key 都有 'module.' 前缀
        # 但我们现在的 'model' 还没被 DDP 包装，所以必须去掉前缀才能匹配
        new_state_dict = {}
        for k, v in state_dict.items():
            # 如果 key 以 'module.' 开头，就去掉前 7 个字符
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # 3. 加载权重到模型
        model.load_state_dict(new_state_dict, strict=False)
        
        if 'rank' in locals() and rank == 0:
            print("权重加载完成，准备开始 Epoch 2 训练。")
    else:
        if 'rank' in locals() and rank == 0:
            print(f"!!! 警告: 找不到 {CHECKPOINT_PATH}，将从头开始训练 !!!")
    # =================================================================

    # ... (接着是 model.to(device) 和 model = DDP(...) ) ...
    
    # [!! 新 !!] 将模型包装在 DDP 中
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    if is_main_process:
        # 注意：在 DDP 中，我们打印 model.module 的参数量
        print(f"模型参数量: {sum(p.numel() for p in model.module.parameters() if p.requires_grad):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler('cuda')
    
    # [!! 新 !!] 添加学习率调度器 (这就是为什么我们需要 transformers)
    total_steps = (len(train_dataset) // (MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS * dist.get_world_size())) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 1000, # 1000 步热身 (现在会很快)
        num_training_steps = total_steps
    )
    
    if is_main_process:
        print("模型、优化器、GradScaler 和调度器已准备就绪。")

    # --- E. 训练循环 ---
    if is_main_process:
        print("--- 4. 开始训练 ---")
    
    for epoch in range(EPOCHS):
        model.train()
        # [!! 新 !!] DDP 必须在每个 epoch 开始时设置 sampler
        train_sampler.set_epoch(epoch)
        
        if is_main_process:
            print(f"\n--- Epoch {epoch + 1} / {EPOCHS} ---")
        
        optimizer.zero_grad() 
        
        for i, batch in enumerate(train_loader):
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(tokens, repr_layers=[30])
                logits = outputs['logits'] 
                loss = loss_fn(
                    logits.view(-1, alphabet_size), 
                    labels.view(-1)
                )
                loss = loss / GRAD_ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            
            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scheduler.step() # [!! 新 !!] 更新学习率
                scaler.update()
                optimizer.zero_grad()
                
                # [!! 新 !!] 只让主进程打印日志
                if is_main_process:
                    current_step = (i + 1) // GRAD_ACCUMULATION_STEPS
                    if current_step % 100 == 0: # 每 100 个 *有效* 批次
                        total_batches = len(train_loader) // GRAD_ACCUMULATION_STEPS
                        print(f"  Epoch {epoch+1}, 有效批次 {current_step}/{total_batches}, "
                              f"MLM 损失: {loss.item() * GRAD_ACCUMULATION_STEPS:.4f}")
            
            # [!! 新 !!] 只让主进程保存检查点
            if is_main_process and (i + 1) % (1000 * GRAD_ACCUMULATION_STEPS) == 0: 
                current_step = (i + 1) // GRAD_ACCUMULATION_STEPS
                print("正在保存临时检查点...")
                # [!! 新 !!] DDP 保存模型时，我们保存 model.module.state_dict()
                torch.save(model.module.state_dict(), os.path.join(SAVE_CHECKPOINT_PATH, f"checkpoint_epoch{epoch}_step{current_step}.pt"))

    if is_main_process:
        print("\n--- 训练完成 ---")
        print("正在保存最终模型...")
        torch.save(model.module.state_dict(), os.path.join(SAVE_CHECKPOINT_PATH, "final_model.pt"))
        print(f"最终模型已保存到: {os.path.join(SAVE_CHECKPOINT_PATH, 'final_model.pt')}")

if __name__ == "__main__":
    main()