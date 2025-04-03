# dataloader
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.core.datamodule import LightningDataModule
from sklearn.metrics import f1_score

# 为了可以用同一个环境设置（sh文件），将父目录加进来
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class HooktheoryNPZDataset(Dataset):
    """
    读取 train_preprocessing_hooktheory.py 产生的 <uid>.npz 文件,
    里头包含: { ..., chunks_features=[1, T, feats_dim], ... }.
    只使用 chunks_features 做训练特征, 忽略 'chunks_tertiaries'等object键.
    """
    def __init__(self, index_file, npz_dir, max_len=384, mode="train"):
        """
        index_file: 列出本 dataset 的 uid (一行一个)
        npz_dir:    存放 .npz 文件的目录
        max_len:    每个样本最多保留多少帧(截断), 也可做padding
        """
        self.npz_dir = npz_dir
        self.max_len = max_len
        self.mode = mode
        self.all_chunks = []    # 存 (uid, chunk_index)

        with open(index_file,"r",encoding="utf-8") as f:
            uids = [line.strip() for line in f if line.strip()]

        # 扫描每个 uid，对它的 .npz 文件中所有 chunk 建立索引
        for uid in uids:
            if self.mode == "train":
                path = os.path.join(self.npz_dir, "valid", f"{uid}.npz")
            elif self.mode == "valid": 
                path = os.path.join(self.npz_dir, "valid", f"{uid}.npz")
            elif self.mode == "test":
                path = os.path.join(self.npz_dir, "test", f"{uid}.npz")
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            if not os.path.isfile(path):
                # raise FileNotFoundError(f"Cannot find {path}")
                continue


            # path = "/scratch/dw3180/sheetsage_project/output/preprocessed_output/train/-Wegl_nKmrY.npz"

            data = np.load(path, allow_pickle=True)

            
            # # 如果没有 chunks_labels_harmony 就跳过
            # if "chunks_labels_harmony" not in data:
            #     print(f"[WARN] uid={uid} has no 'chunks_labels_harmony'. Skipping.")
            #     continue
            # if "chunks_features_mert" not in data:
            #     print(uid)
            #     print(f"[WARN] uid={uid} has no 'chunks_features_mert'. Skipping.")
            #     continue
            
            feats = data["chunks_features"]

            N_chunks = feats.shape[0]
            # 把 (uid, c) 全部存下来，表示第 c 个 chunk
            for c in range(N_chunks):
                self.all_chunks.append((uid, c))

    def __len__(self):
        return len(self.all_chunks)

    def __getitem__(self, idx):
        uid, c = self.all_chunks[idx]
        # print(f"[DEBUG] idx={idx}, uid={uid} -> before loading")
        if self.mode == "train":
            path = os.path.join(self.npz_dir, "valid", f"{uid}.npz")
        elif self.mode == "valid":
            path = os.path.join(self.npz_dir, "valid", f"{uid}.npz")
        

        # path = "/scratch/dw3180/sheetsage_project/output/preprocessed_output/train/-Wegl_nKmrY.npz"

        # 允许 pickle=True, 避免 dtype=object 读取报错; 但我们只关心 chunks_features
        data = np.load(path, allow_pickle=True)

        # feats
        feats = data["chunks_features"]  # shape => (N_chunks, T, feats_dim)
        chunk = feats[c]  # shape => [T, feats_dim]
        T = chunk.shape[0]

        labels = data["chunks_labels"]  # shape => (N_chunks, T) 或 object array
        # labels = data["chunks_labels_harmony"]
        y_chunk = labels[c]

        # 对 label 做 +28 平移
        # 先转成 int64 以防出现其他类型
        y_chunk = y_chunk.astype(np.int64)  
        y_chunk = y_chunk + 28             # melody所有标签直接加 28
        min_val2 = y_chunk.min()
        max_val2 = y_chunk.max()
        # print(f"[DEBUG] after offset: min={min_val2}, max={max_val2}")


        # # 1) 如果你想限制最大长度，可以裁剪
        # if T > self.max_len:
        #     chunk = chunk[:self.max_len]
        #     y_chunk = y_chunk[:self.max_len] # labels
        #     T = self.max_len

        # # 2) pad 到固定长度
        # if T < self.max_len:
        #     pad_len = self.max_len - T
        #     chunk = np.pad(chunk, [(0, pad_len), (0, 0)])  # shape -> [max_len, feats_dim]
        #     y_chunk = np.pad(y_chunk, (0, pad_len))        # => [max_len]


        # 3) 转成 tensor
        # chunk_t = torch.tensor(chunk, dtype=torch.float32)

        # 4) label, 这里只是 dummy
        # 如果是做自回归预测，可能 label.shape = [max_len] ...
        # y_chunk = np.zeros(len(y_chunk), dtype=np.int64)
        # y = torch.zeros(T, dtypee=torch.long)

        return chunk, y_chunk
    

def collate_fn(batch):
    """
    目的：针对 batch 中不同样本帧数（T）不一致的问题，动态 pad 到一个统一长度
    batch: list of (feats_np, y_np)
    feats_np => shape [T, feats_dim], y_np => shape [T].
    不同样本T可不同 => 需pad
    """
    import torch.nn.functional as F

    max_T = max(b[0].shape[0] for b in batch)  # batch中最大T
    print('max T:', max_T)
    feats_dim = batch[0][0].shape[1]  # feats_dim

    feats_list = []
    ys_list = []
    src_len_list = []

    for i, (feats_np, y_np) in enumerate(batch):
        T = feats_np.shape[0]
        # print('T:', T)
        src_len_list.append(T)
        # 先转tensor
        feats_t = torch.from_numpy(feats_np).float()  # shape [T, feats_dim]
        y_t     = torch.from_numpy(y_np).long()       # shape [T]

        if T < max_T:
            # pad
            pad_len = max_T - T
            # feats_t shape [T, feats_dim],  2D pad => ( (dim2_left, dim2_right), (dim1_left, dim1_right) ) in PyTorch
            # 这里是 -> [T, feats_dim], 只能对 2D (N,H).
            # param= (0,0,0,pad_len) => " (left, right, top, bottom) " for 2D?
            pad_f = F.pad(feats_t, (0, 0, 0, pad_len)) # (pad_w_left, pad_w_right, pad_h_left, pad_h_right)
            pad_y = F.pad(y_t, (0, pad_len))
        else:
            pad_f = feats_t
            pad_y = y_t

        # print(f"[DEBUG] after pad: pad_f.shape={pad_f.shape}, pad_y.shape={pad_y.shape}")

        feats_list.append(pad_f)
        ys_list.append(pad_y)

    # stack => [B, max_T, feats_dim], [B, max_T]
    feats_batch = torch.stack(feats_list, dim=0)
    ys_batch = torch.stack(ys_list, dim=0)
    feats_batch = feats_batch.transpose(0, 1)  # (0,1) => (1,0) 结果是 [max_T, B, feats_dim]
    ys_batch = ys_batch.transpose(0, 1)        # 如果你的 label 也要匹配 [T, B]
    # print('feats batch', feats_batch)
    # print('ys batch', ys_batch)
    src_len = torch.tensor(src_len_list, dtype=torch.long)

    return (feats_batch, src_len, ys_batch)


import pytorch_lightning as pl
from torch.utils.data import DataLoader

class HooktheoryDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.get("batch_size", 16)
        self.max_len = cfg.get("src_max_len",384)
        # 下面 index_file之类路径, 你可写在cfg
        self.train_index = cfg["train_index"]  # e.g. "train_uids.txt"
        self.val_index   = cfg["val_index"]    # e.g. "val_uids.txt"
        self.test_index  = cfg["test_index"]    # e.g. "test_uids.txt"
        self.npz_dir     = cfg["npz_dir"]      # e.g. "/sheetsage/output/preprocessed_output"

    def setup(self, stage=None):
        self.train_dataset = HooktheoryNPZDataset(
            index_file=self.train_index,
            npz_dir=self.npz_dir,
            max_len=self.max_len,
            mode="train"   # 显式地指定训练模式
        )
        self.val_dataset = HooktheoryNPZDataset(
            index_file=self.val_index,
            npz_dir=self.npz_dir,
            max_len=self.max_len,
            mode="valid"   # 显式地指定验证模式
        )
        self.test_dataset = HooktheoryNPZDataset(
            index_file=self.test_index,
            npz_dir=self.npz_dir,
            max_len=self.max_len,
            mode="test"   # 显式地指定验证模式
        )
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0   # 加快数据加载
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )


# model_lightning.py
import torch.nn.functional as F
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint

class SheetsageLightningModule(pl.LightningModule):
    """
    这个LightningModule将包含:
    - __init__: 构建模型(相当于 create_model ), 接收 cfg
    - forward: 只写推理过程
    - training_step, validation_step: 替代 compute_loss / compute_eval_metrics 
    - configure_optimizers: 替代 train.py 里创建 optimizer, scheduler
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 1) 构建网络 (相当于create_model逻辑)
        #    需要 from modules import EncOnlyTransducer, TransformerEncoder
        from sheetsage.modules import EncOnlyTransducer, TransformerEncoder, SimpleEncoderOnlyTransformer

        # 一些映射:
        task_to_vocab_size = {"melody": 89, "harmony": 97}
        task_name = cfg.get("task", "melody")
        self.output_dim = task_to_vocab_size.get(task_name, 89)

        input_feats_to_dim = {"HANDCRAFTED": 229, "JUKEBOX": 4800, "MERT": 768}
        feats_name = cfg.get("input_feats", "JUKEBOX")
        src_dim = input_feats_to_dim.get(feats_name, 4800)

        model_type = cfg.get("model", "transformer")
        hacks = cfg.get("hacks", [])

        if model_type == "transformer":
            self.model = EncOnlyTransducer(
                output_dim=self.output_dim,
                src_emb_mode="project",
                src_dim=src_dim,
                src_emb_dim=512,
                src_pos_emb=("pos_emb" in hacks),
                src_dropout_p=0.1,
                enc_cls=TransformerEncoder,
                enc_kwargs={
                    "model_dim": 512,
                    "num_heads": 8,
                    "num_layers": 4 if "4layers" in hacks else 6,
                    "feedforward_dim": 2048,
                    "dropout_p": 0.1,
                }
            )
        else:
            raise NotImplementedError(f"Unsupported model type {model_type}")
        # self.model = SimpleEncoderOnlyTransformer(
        #     src_dim=src_dim,
        #     output_dim=self.output_dim,
        #     model_dim=256,
        #     num_heads=4,
        #     num_layers=3,
        #     feedforward_dim=1024,
        #     dropout_p=0.1,
        #     use_pos_emb=True,  # 你想要可学习位置编码就True，否则False
        # )

    def forward(self, src, src_len):
        """
        LightningModule的 forward, 只写推理逻辑
        """
        logits = self.model(src, src_len)
        return logits

    def training_step(self, batch, batch_idx):
        """
        训练时每个batch调用一次:
        batch通常是 (src, src_len, y)
        这里做 forward => loss => log => return
        """
        src, src_len, ys = batch
        # print('ys', ys)
        # print('ys shape', ys.shape)
        logits = self(src, src_len)  # shape [T,B,out_dim]
        loss_val = self._compute_loss(logits, ys, src_len)

        # logging
        # self.logger.experiment.log_metrics({"train_loss": loss_val}, step=self.global_step)
        # 1) 直接使用Lightning的 self.log 来记录训练loss
        #    on_step=True 表示每一步都记录; on_epoch=True 表示每个epoch结束汇总.
        self.log("train_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss_val

    def validation_step(self, batch, batch_idx):
        """
        验证/评估时每个batch调用
        计算acc或loss
        """
        src, src_len, ys = batch
        logits = self(src, src_len)
        loss_val = self._compute_loss(logits, ys, src_len)
        acc = self._compute_acc(logits, ys, src_len)
        f1 = self._compute_f1(logits, ys, src_len)
        # self.log 用于Lightning的tensorboard/wandb记录
        # self.logger.experiment.log_metrics({"val_loss": loss_val, "val_acc": acc}, step=self.global_step)

        # 2) 直接使用Lightning的 self.log 来记录验证loss和acc
        self.log("val_loss", loss_val, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc,       on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", f1,       on_epoch=True, prog_bar=True, logger=True)

        # 如果想后续 cat，它们需要在相同的第一个维度上进行拼接；我们可以将 batch 维放在第 1 维。可自行调整。
        return {
            "val_loss": loss_val,
            "val_acc": acc,
            "val_f1": f1,
            "logits": logits.detach(),  # 不要忘记 detach
            "ys": ys.detach(),
            "src_len": src_len.detach()  # 存储 src_len
        }
    
    def validation_epoch_end(self, outputs):
        """
        收集所有 batch 的 logits 和 ys，并进行保存或其他处理
        outputs: list of dicts, 每个元素是 validation_step 的返回值
        logits 是大小为 [T,B,C] 的连续数值（可能是任意浮点数），表示对每个时间步、每个样本、每个类别的预测分数。
        ys 是大小为[T,B] 的整数序列，每个元素都在[0,C) 范围内（即某个类别 ID）。
        """
        # 把所有 batch 的 logits、ys 拼接
        # 找到所有 batch 中的最大 T
        max_T = max([o["logits"].shape[0] for o in outputs])
        print('真max', max_T)
        # 对每个 batch 进行 padding，使 T 维度对齐到 max_T
        padded_logits = []
        padded_ys = []
        all_src_len = []

        for o in outputs:
            logits = o["logits"]  # [T, B, out_dim]
            ys = o["ys"]          # [T, B]
            src_len = o["src_len"]  # [B]

            # 计算需要 pad 的数量
            pad_T = max_T - logits.shape[0]

            # 执行 padding
            padded_logits.append(torch.nn.functional.pad(logits, (0, 0, 0, 0, 0, pad_T)))  # 在T维进行pad
            padded_ys.append(torch.nn.functional.pad(ys, (0, 0, 0, pad_T)))  # 在T维进行pad
            all_src_len.append(src_len)
            
        print('padded logits', padded_logits[0].shape, padded_logits[1].shape)
        print('padded ys', padded_ys[0].shape, padded_ys[1].shape)
        # 将所有 batch 的 logits 和 ys 拼接
        all_logits = torch.cat(padded_logits, dim=1)  # [max_T, sum_of_B, out_dim]
        all_ys     = torch.cat(padded_ys, dim=1)      # [max_T, sum_of_B]
        all_src_len = torch.cat(all_src_len, dim=0)  # [sum_of_B]

        # 确保保存的时候维度匹配
        assert all_logits.shape[1] == all_ys.shape[1] == all_src_len.shape[0], \
        f"Mismatch in dimensions: logits={all_logits.shape}, ys={all_ys.shape}, src_len={all_src_len.shape}"

        # 如果需要在 CPU 上保存，可以先 .cpu()
        # all_logits = all_logits.cpu()
        # all_ys = all_ys.cpu()

        # 这里演示把结果保存到一个 .pt 文件，可以按需自定义路径、文件名
        # 例如按 epoch 命名。下面示例保存在当前目录下。
        save_dir = "val_outputs"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "logits": all_logits,
                "ys": all_ys,
                "src_len": all_src_len
            },
            os.path.join(save_dir, f"Juke_mel_refinedSub_TVsame_mask_val_epoch_{self.current_epoch}.pt")
        )
        self.print(f"[INFO] 验证集 logits/ys 已保存: {os.path.join(save_dir, f'val_epoch_{self.current_epoch}.pt')}")


    def configure_optimizers(self):
        """
        替代train.py里的优化器, 
        也可return scheduler, multiple optim等
        """
        lr = self.cfg.get("lr", 1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    # 下面是helper:
    def _compute_loss(self, logits, ys, src_len):
        # logits: [T,B,out_dim], ys: [T,B]
        # 1) 先展开到 2D
        #    注意 logit 维度顺序 [T, B, C]，我们 reshape 前最好先把 B 放前面
        logits_2d = logits.reshape(-1, self.output_dim)  # [T*B, out_dim]
        ys_2d = ys.reshape(-1)                           # [T*B]
        
        # 检查标签 min/max
        min_y = ys_2d.min().item()
        max_y = ys_2d.max().item()
        if min_y < 0 or max_y >= self.output_dim:
            raise ValueError(f"Label out of range: min={min_y}, max={max_y},"
                            f" output_dim={self.output_dim}")

        # 2) 用 length 构造 "valid_mask"
        #    seq_idxs[i,j] < src_len[i] 就表示这是有效帧；否则是 padding
        print('logits shape', logits.shape)
        print('ys shape', ys.shape)
        B = ys.shape[1]
        T = ys.shape[0]
        seq_idxs = torch.arange(0, T, dtype=src_len.dtype, device=ys.device).expand(B, -1)  # [B,T]
        pad_mask = seq_idxs >= src_len.unsqueeze(1)               # True表示“padding”
        pad_mask = pad_mask.transpose(0,1)
        valid_mask = ~pad_mask                                    # True表示“有效帧”

        # 3) 同样 reshape 到 [B*T]
        valid_mask_1d = valid_mask.reshape(-1)

        # 4) 先计算逐元素的 CrossEntropy
        loss_per_pos = F.cross_entropy(
            logits_2d,
            ys_2d,
            reduction='none'  # 不要 ignore_index
        )  # [T,B]

        # 5) 把padding处的loss清零
        loss_per_pos = loss_per_pos * valid_mask_1d

        # 6) 在有效帧上做平均
        denom = valid_mask_1d.sum().clamp(min=1)  # 避免除0
        loss_val = loss_per_pos.sum() / denom
        # loss_val = F.cross_entropy(
        #     logits_2d,
        #     ys_2d,
        # ) 
        return loss_val

    def _compute_acc(self, logits, ys, src_len):
        # logits: [T, B, out_dim], ys: [B, T]
        preds = logits.argmax(dim=-1)  # [T, B]

        # 用 valid_mask 过滤掉 padding 的部分
        B = ys.shape[1]
        T = ys.shape[0]
        seq_idxs = torch.arange(0, T, dtype=src_len.dtype, device=ys.device).expand(B, -1)  # [B,T]
        pad_mask = seq_idxs >= src_len.unsqueeze(1)               # True表示“padding”
        pad_mask = pad_mask.transpose(0,1)
        valid_mask = ~pad_mask                                    # True表示“有效帧”

        preds_flat = preds[valid_mask].reshape(-1)  # 只考虑有效帧的预测结果
        ys_flat = ys[valid_mask].reshape(-1)        # 只考虑有效帧的真实标签

        # preds_flat = preds.reshape(-1)
        # ys_flat = ys.reshape(-1)
        acc = (preds_flat == ys_flat).float().mean()
        # correct = (preds_flat == ys_flat).sum()
        # total = ys.numel()
        # acc = correct.float() / total if total > 0 else 0.0
        return acc
    

    def _compute_f1(self, logits, ys, src_len):
        preds = logits.argmax(dim=-1)  # [T, B]

        # 用 valid_mask 过滤掉 padding 的部分
        B = ys.shape[1]
        T = ys.shape[0]
        seq_idxs = torch.arange(0, T, dtype=src_len.dtype, device=ys.device).expand(B, -1)  # [B,T]
        pad_mask = seq_idxs >= src_len.unsqueeze(1)               # True表示“padding”
        pad_mask = pad_mask.transpose(0,1)
        valid_mask = ~pad_mask                                    # True表示“有效帧”

        preds_flat = preds[valid_mask].reshape(-1).cpu().numpy()  # 只考虑有效帧的预测结果
        ys_flat = ys[valid_mask].reshape(-1).cpu().numpy()        # 只考虑有效帧的真实标签

        f1 = f1_score(ys_flat, preds_flat, average='micro', zero_division=0)  
        return f1


# main_lightning.py
import logging
import json

class Config:
    def __init__(self, d):
        self.d = d
        if "batch_size" not in self.d:
            self.d["batch_size"] = 16
        if "lr" not in self.d:
            self.d["lr"] = 1e-3
        if "src_max_len" not in self.d:
            self.d["src_max_len"] = 384
        if "seed" not in self.d:
            self.d["seed"] = 42
        if "epochs" not in self.d:
            self.d["epochs"] = 10
        if "train_index" not in self.d:
            self.d["train_index"] = "train_uids.txt"
        if "val_index" not in self.d:
            self.d["val_index"] = "val_uids.txt"
        if "npz_dir" not in self.d:
            self.d["npz_dir"] = "/sheetsage/output/preprocessed_output_refined"
    def get(self, k, default=None):
        return self.d.get(k, default)


def main():
    if torch.cuda.is_available():
        print("当前 GPU id:", torch.cuda.current_device())
        print("GPU 名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("已分配的 GPU 内存:", torch.cuda.memory_allocated())
    
    # 1) 加载cfg
    cfg_path = "/home/dw3180/sheetsage_code/calmdown/config_sub.json"
    # cfg_path = "/home/dw3180/sheetsage_code/sheetsage/config.json"
    with open(cfg_path,"r") as f:
        d = json.load(f)
    cfg = Config(d)

    # 2) 设置seed
    pl.seed_everything(cfg.get("seed",42))

    # 3) 构建LightningModule
    model = SheetsageLightningModule(cfg.d)  # .d or pass entire config

    # 4) 构建DataLoader
    data_module = HooktheoryDataModule(cfg.d)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",        # 跟踪 val_loss
        dirpath="lightning_logs/real_refinedSub_TVsame_mask/checkpoints",
        filename="Juke_melody_epoch={epoch}-val_loss={val_loss:.4f}",
        save_top_k=1,              # 只保存最好的1个
        mode="min"                 # val_loss越小越好
    )

    # 5) 构建Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.get("epochs", 10),  # 最大训练轮数
        gpus=1 if torch.cuda.is_available() else 0,  # 使用 1 个 GPU
        checkpoint_callback=checkpoint_callback,
        default_root_dir="./lightning_logs",  # 日志和检查点保存目录
        check_val_every_n_epoch=1, # 每个 epoch 做一次 val
        progress_bar_refresh_rate=10,  # 进度条刷新频率
    )

    # 6) trainer.fit
    trainer.fit(model, datamodule=data_module)

if __name__=="__main__":
    main()
