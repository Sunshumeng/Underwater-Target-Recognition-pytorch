import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from tqdm import tqdm
from dataload import SoundDataSet, get_dataloader_with_test


# ================= Label Smoothing =================
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        num_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


# ================== 参数 ==================
def read_classes_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def get_args():
    parser = argparse.ArgumentParser(description='Audio classification (accuracy + feature consistency)')
    parser.add_argument(
        '-t', type=str,
        default='pytorch-audioclassification-master-main-MT',
        help="theme name"
    )
    parser.add_argument(
        '-dp', type=str,
        default=r'D:\音频分类\Pytorch-AudioClassification-master-main - MT\data\scatter\npy_data',
        help="train data directory"
    )
    parser.add_argument(
        '-classes', type=str,
        default=r'D:\音频分类\Pytorch-AudioClassification-master-main - MT\data\scatter\classes.txt',
        help="Path to the classes file"
    )
    parser.add_argument(
        '-infop', type=str,
        default=r'D:\音频分类\Pytorch-AudioClassification-master-main - MT\data\scatter\refer.csv',
        help="CSV meta file path"
    )
    parser.add_argument(
        '-bs', type=int,
        default=8,
        help="batch size"
    )
    parser.add_argument(
        '-cn', type=int,
        default=32,
        help='number of classes'
    )
    parser.add_argument(
        '-e', type=int,
        default=100,
        help='epochs'
    )
    parser.add_argument(
        '-lr', type=float,
        default=0.001,
        help='learning rate'
    )
    parser.add_argument(
        '-ld', type=str,
        default=r'D:\音频分类\Pytorch-AudioClassification-master-main - MT\workdir',
        help="log save directory"
    )
    parser.add_argument(
        '-model_type', type=str,
        default='cnn',
        choices=['cnn'],
        help="model type"
    )
    return parser.parse_args()

USE_SPEC_AUG = False
AUG_LAST_EPOCHS = 10


def spec_augment(batch_spectrogram,
                 time_mask_ratio=0.1,
                 freq_mask_ratio=0.1,
                 num_masks=2):
    x = batch_spectrogram.clone()
    B, C, F_, T_ = x.shape
    device = x.device

    for i in range(B):
        xi = x[i]

        for _ in range(num_masks):
            t = int(T_ * time_mask_ratio * torch.rand(1, device=device).item())
            if t > 0:
                t0 = torch.randint(0, max(1, T_ - t), (1,), device=device).item()
                xi[:, :, t0:t0 + t] = 0.0

        for _ in range(num_masks):
            f = int(F_ * freq_mask_ratio * torch.rand(1, device=device).item())
            if f > 0:
                f0 = torch.randint(0, max(1, F_ - f), (1,), device=device).item()
                xi[:, f0:f0 + f, :] = 0.0

        x[i] = xi

    return x


# ================== 主函数 ==================
def main():
    args = get_args()

    # ===== 设备 =====
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # ===== 数据与类别 =====
    data_dir = args.dp
    csv_path = args.infop
    class_path = args.classes
    classes = read_classes_from_file(class_path)

    dataset = SoundDataSet(data_dir, csv_path)
    labels_all = dataset.df['label'].values
    print("整体类别分布：")
    print(pd.Series(labels_all).value_counts())

    train_loader, valid_loader, test_loader, _ = get_dataloader_with_test(
        data_dir=data_dir,
        csv_path=csv_path,
        batch_size=args.bs,
        train_percent=0.6,
        valid_percent=0.2,
        num_workers=4,
        random_state=42,
        use_stratify=True,
    )

    # ===== 模型 =====
    from model.cnn import AudioClassificationModelCNN
    model = AudioClassificationModelCNN(num_classes=args.cn).to(device)

    for name, param in model.named_parameters():
        if "conv1" in name or "conv2" in name:
            param.requires_grad = False

    print("\n模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    loss_fn_cls = LabelSmoothingCE(smoothing=0.1)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    exp_dir = os.path.join(args.ld, f"exp_accfocus_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_f1_path = os.path.join(checkpoints_dir, 'best_f1.pth')

    best_f1 = 0.0
    best_acc = 0.0
    best_conf_mat = None

    train_loss_curve = []
    val_loss_curve = []

    alpha = 1.0
    beta_cons = 0.02

    # ===================== 训练循环 =====================
    for epoch in range(args.e):

        if epoch == 5:
            print("\n>>> Unfreeze encoder for fine-tuning <<<\n")
            for param in model.parameters():
                param.requires_grad = True

            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=1e-4
            )
            scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

        # ---------- Train ----------
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.e}', ncols=90)

        for batch in train_bar:
            data, labels = batch
            inputs = data.to(device).float()
            labels = labels.to(device)

            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            clean_inputs = inputs

            if USE_SPEC_AUG and epoch >= args.e - AUG_LAST_EPOCHS:
                aug_inputs = spec_augment(inputs)
            else:
                aug_inputs = inputs

            optimizer.zero_grad()

            outputs = model(aug_inputs)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits, _ = outputs
            else:
                logits = outputs

            # 分类损失
            loss_cls = loss_fn_cls(logits, labels)

            feat_clean = model.extract_features(clean_inputs)
            feat_aug = model.extract_features(aug_inputs)
            feat_clean = F.normalize(feat_clean, dim=1)
            feat_aug = F.normalize(feat_aug, dim=1)
            loss_cons = F.mse_loss(feat_clean, feat_aug)

            loss = alpha * loss_cls + beta_cons * loss_cons

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            train_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Cls": f"{loss_cls.item():.4f}",
                "Cons": f"{loss_cons.item():.4f}",
            })

        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_curve.append(avg_train_loss)
        scheduler.step()

        # ---------- Validation ----------
        model.eval()
        all_preds, all_labels = [], []
        val_epoch_loss = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                data, labels = batch
                inputs = data.to(device).float()
                labels = labels.to(device)

                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)

                clean_inputs = inputs
                aug_inputs = spec_augment(inputs) if USE_SPEC_AUG else inputs

                outputs = model(clean_inputs)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, _ = outputs
                else:
                    logits = outputs

                loss_cls = loss_fn_cls(logits, labels)

                feat_clean = model.extract_features(clean_inputs)
                feat_aug = model.extract_features(aug_inputs)
                feat_clean = F.normalize(feat_clean, dim=1)
                feat_aug = F.normalize(feat_aug, dim=1)
                loss_cons = F.mse_loss(feat_clean, feat_aug)

                val_loss = alpha * loss_cls + beta_cons * loss_cons
                val_epoch_loss += val_loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_epoch_loss / len(valid_loader)
        val_loss_curve.append(avg_val_loss)

        # 验证集指标
        val_acc = accuracy_score(all_labels, all_preds)
        val_pre = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"\nEpoch {epoch + 1} Summary:")
        tab = PrettyTable(['TrainLoss', 'ValLoss', 'Val-Acc', 'Val-Prec', 'Val-Rec', 'Val-F1'])
        tab.add_row([avg_train_loss, avg_val_loss, val_acc, val_pre, val_rec, val_f1])
        print(tab)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_f1_path)
            print(f"  -> New best F1 on valid: {best_f1:.4f} (saved)")

        if val_acc > best_acc:
            best_acc = val_acc
            best_conf_mat = confusion_matrix(all_labels, all_preds)

    # ===== 最佳验证集混淆矩阵====
    if best_conf_mat is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(best_conf_mat, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Best Valid Confusion Matrix (Acc={best_acc:.2%})")
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "best_valid_confusion_matrix.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(train_loss_curve, label='Train Loss', linewidth=2)
        plt.plot(val_loss_curve, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        loss_fig_path = os.path.join(exp_dir, "loss_curve.png")
        plt.savefig(loss_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss 曲线已保存至: {loss_fig_path}")

    # ===== 测试集评估（使用 best F1 权重）=====
    model.load_state_dict(torch.load(best_f1_path, map_location=device))
    model.eval()

    all_test_preds, all_test_labels = [], []
    correct_top3 = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            data, labels = batch
            inputs = data.to(device).float()
            labels = labels.to(device)

            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits, _ = outputs
            else:
                logits = outputs

            preds = torch.argmax(logits, dim=1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

            topk = 3
            _, topk_idx = logits.topk(topk, dim=1)
            correct_top3 += (topk_idx == labels.view(-1, 1)).any(dim=1).sum().item()
            total_samples += labels.size(0)

    test_acc = accuracy_score(all_test_labels, all_test_preds)
    test_pre = precision_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
    test_rec = recall_score(all_test_labels, all_test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_test_labels, all_test_preds, average='weighted')
    top3_acc = correct_top3 / max(total_samples, 1)

    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Precision: {test_pre:.4f}")
    print(f"Test Recall   : {test_rec:.4f}")
    print(f"Test F1 Score : {test_f1:.4f}")
    print(f"Test Top-3 Acc: {top3_acc:.4f}")

    test_conf = confusion_matrix(all_test_labels, all_test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_conf, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Test Confusion Matrix (Acc={test_acc:.2%})")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "test_confusion_matrix.png"), dpi=300)
    plt.close()

    final_model_path = os.path.join(checkpoints_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n模型已保存至: {final_model_path}")


if __name__ == "__main__":
    main()

