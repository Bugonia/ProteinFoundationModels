import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pyfaidx

# --- 1. 从您提供的文件中导入 ---
try:
    from .data import Alphabet
    from .esm2 import ESM2
except ImportError:
    print("错误: 确保 data.py 和 esm2.py 在同一目录中。")
    print("您是否从父目录使用了 'python -m esm.esm.train_... '？")
    sys.exit(1)

# ----------------------------------------------------------------------
# 2. 高效的 Fasta 数据集
# ----------------------------------------------------------------------
class FastaDataset(Dataset):
    def __init__(self, fasta_file):
        print(f"正在打开 FASTA 文件: {fasta_file}")
        self.fasta = pyfaidx.Fasta(fasta_file)
        self.keys = list(self.fasta.keys())
        print(f"成功加载 FASTA。找到 {len(self.keys)} 条序列。")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        seq_id = self.keys[idx]
        seq = str(self.fasta[seq_id])
        return seq

# ----------------------------------------------------------------------
# 3. [!! 关键 !!] MLM Data Collator (掩码器)
# ----------------------------------------------------------------------
class MLMDataCollator:
    def __init__(self, alphabet: Alphabet, max_len: int, mlm_probability: float = 0.15): # <--- 接收 max_len
        self.alphabet = alphabet
        self.max_len = max_len # <--- 使用传入的 max_len
        self.mlm_probability = mlm_probability
        self.pad_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.prob_mask = mlm_probability * 0.8
        self.prob_random = mlm_probability * 0.1

    def __call__(self, batch_seqs: list[str]):
        tokenized_seqs = [self.alphabet.encode(seq)[:self.max_len] for seq in batch_seqs] # <--- 截断
        
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
        
        # [!! 新 !!] 确保填充后的序列不超过 max_len
        # (如果批次中最长的序列短于 max_len，这没有影响)
        # (如果批次中最长的序列长于 max_len, 它会被 pad_sequence 填充到该长度)
        # (但如果所有序列都短于 max_len, 我们可以节省内存)
        current_max_len = padded_tokens.size(1)
        if current_max_len > self.max_len:
            padded_tokens = padded_tokens[:, :self.max_len]
            padded_labels = padded_labels[:, :self.max_len]
        
        return {"tokens": padded_tokens, "labels": padded_labels}

# ----------------------------------------------------------------------
# 4. 主训练脚本 (!! 已更新 !!)
# ----------------------------------------------------------------------
def main():
    
    # --- A. 设置 (!! 关键路径 !!) ---
    print("--- 1. 开始设置 [fair-esm 预训练] ---")
    
    FASTA_FILE_PATH = "/inspire/ssd/project/advanced-machine-learning-and-deep-learning-applications/zhongxiaoqiu-253108120179/data/uniref50/uniref50.fasta"
    SAVE_CHECKPOINT_PATH = "/inspire/hdd/global_user/zhongxiaoqiu-253108120179/models/esm2_150M_fair_esm_run"
    
    # --- B. [!! 新 !!] 训练参数 ---
    #
    # 内存管理：
    # 有效批量大小 = MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS
    # ( 16 * 8 = 128 )
    #
    MICRO_BATCH_SIZE = 16     # [!! 新 !!] 这是实际放入 GPU 的批量大小。16 应该安全。
    GRAD_ACCUMULATION_STEPS = 8 # [!! 新 !!] 累积多少个 micro-batch 再更新一次权重。
    
    # [!! 关键 !!] 将 max_len 从 1024 减半。这将使内存占用减少 4 倍！
    MAX_LEN = 512
    
    EPOCHS = 1
    LEARNING_RATE = 5e-5
    NUM_WORKERS = 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} ({torch.cuda.get_device_name(0)})")
    os.makedirs(SAVE_CHECKPOINT_PATH, exist_ok=True)
    
    # --- C. 加载 Tokenizer (Alphabet) 和数据 ---
    print("--- 2. 加载数据 ---")
    
    alphabet = Alphabet.from_architecture("ESM-1b")
    alphabet_size = len(alphabet)
    
    train_dataset = FastaDataset(FASTA_FILE_PATH)
    
    # [!! 已更新 !!] 传入 MAX_LEN
    data_collator = MLMDataCollator(alphabet, max_len=MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICRO_BATCH_SIZE, # <--- [!! 已更新 !!] 使用 Micro Batch
        shuffle=False,
        collate_fn=data_collator,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"数据加载器已准备就绪。共 {len(train_dataset)} 条序列。")
    print(f"有效批量大小: {MICRO_BATCH_SIZE * GRAD_ACCUMULATION_STEPS} (Micro-Batch: {MICRO_BATCH_SIZE} * {GRAD_ACCUMULATION_STEPS} 步)")
    print(f"最大序列长度: {MAX_LEN}")
    
    # --- D. 初始化模型、损失和优化器 ---
    print("--- 3. 初始化模型 (从头开始) ---")
    
    model = ESM2(
        num_layers=30,
        embed_dim=640,
        attention_heads=8,
        alphabet=alphabet
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # [!! 新 !!] 初始化 GradScaler 用于自动混合精度 (AMP)
    scaler = torch.amp.GradScaler('cuda')

    print("模型、优化器、GradScaler 已准备就绪。")

    # --- E. 训练循环 (!! 已更新 !!) ---
    print("--- 4. 开始训练 ---")
    model.train()
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1} / {EPOCHS} ---")
        
        # [!! 新 !!] 在循环开始时清零优化器
        optimizer.zero_grad() 
        
        for i, batch in enumerate(train_loader):
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            # [!! 已更新 !!] 开启 AMP (修复了 FutureWarning)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(tokens, repr_layers=[30])
                logits = outputs['logits'] 
                loss = loss_fn(
                    logits.view(-1, alphabet_size), 
                    labels.view(-1)
                )
                
                # [!! 新 !!] 标准化累积的损失
                loss = loss / GRAD_ACCUMULATION_STEPS

            # [!! 已更新 !!] 使用 scaler 进行反向传播
            scaler.scale(loss).backward()
            
            # [!! 新 !!] 梯度累积逻辑
            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                # 累积了足够的梯度，现在执行更新
                scaler.unscale_(optimizer) # Unscale
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 裁剪
                scaler.step(optimizer) # 优化器 step
                scaler.update() # 更新 scaler
                optimizer.zero_grad() # 为下一个累积周期清零

                # 仅在执行更新时打印日志
                if (i + 1) % (100 * GRAD_ACCUMULATION_STEPS) == 0: # 每 100 个 *有效* 批次
                    effective_batch_num = (i + 1) // GRAD_ACCUMULATION_STEPS
                    print(f"  Epoch {epoch+1}, 有效批次 {effective_batch_num}/{len(train_loader) // GRAD_ACCUMULATION_STEPS}, "
                          f"MLM 损失: {loss.item() * GRAD_ACCUMULATION_STEPS:.4f}") # 乘以累积步数以回报
            
            if (i + 1) % (1000 * GRAD_ACCUMULATION_STEPS) == 0: 
                effective_batch_num = (i + 1) // GRAD_ACCUMULATION_STEPS
                print("正在保存临时检查点...")
                torch.save(model.state_dict(), os.path.join(SAVE_CHECKPOINT_PATH, f"checkpoint_epoch{epoch}_batch{effective_batch_num}.pt"))

    print("\n--- 训练完成 ---")
    
    print("正在保存最终模型...")
    torch.save(model.state_dict(), os.path.join(SAVE_CHECKPOINT_PATH, "final_model.pt"))
    print(f"最终模型已保存到: {os.path.join(SAVE_CHECKPOINT_PATH, 'final_model.pt')}")


if __name__ == "__main__":
    main()