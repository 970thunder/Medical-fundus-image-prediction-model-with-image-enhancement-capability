# -*- codeing = utf-8 -*-
# 科 研 小 分 队
# @Author：李鑫
# 3/3/2025 下午6:37
# @Fire : main.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import pytorch_lightning as pl
from timm import create_model
from sklearn.metrics import roc_auc_score, f1_score
import cv2
from pytorch_lightning.callbacks import ModelCheckpoint


# ----------------------
# 1. 数据准备与预处理
# ----------------------
class EyeDataset(Dataset):
    def __init__(self, excel_path, img_root, transform=None, mode='train'):
        self.df = pd.read_excel(excel_path)
        self.img_root = img_root
        self.transform = transform
        self.mode = mode
        self._process_data()

    def _process_data(self):
        # 处理左右眼数据为独立样本
        left_data = self.df[['Left-Fundus', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].copy()
        left_data.columns = ['path', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        right_data = self.df[['Right-Fundus', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].copy()
        right_data.columns = ['path', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

        self.data = pd.concat([left_data, right_data], ignore_index=True)
        self.data = self.data[self.data['path'].notnull()].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_root, row['path'])

        # 检查图片文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片文件 {img_path} 不存在，请检查路径！")

        image = Image.open(img_path).convert('RGB')

        # 多标签处理
        labels = row[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels)


# 医学影像专用数据增强
def get_transforms(phase='train'):
    if phase == 'train':
        return T.Compose([
            T.RandomRotation(15),
            T.RandomResizedCrop(512, scale=(0.8, 1.2)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize(512),
            T.CenterCrop(512),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ----------------------
# 2. 图像增强模块 (Zero-DCE完整实现)
# ----------------------
class DCE_Net(nn.Module):
    def __init__(self, num_iter=8):
        super().__init__()
        self.num_iter = num_iter
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 3, 3, padding=1),
                nn.Tanh()
            ) for _ in range(num_iter)
        ])

    def forward(self, x):
        # 初始化增强图像为原始输入
        enhanced = x.clone()
        for layer in self.conv_layers:
            # 每层输出调整参数，并叠加到增强图像上
            delta = layer(enhanced)
            enhanced = enhanced + delta
        return enhanced


class ZeroDCE(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = DCE_Net()
        self.lr = lr

    def _loss_function(self, enhanced, original):
        # 空间一致性损失
        loss_spatial = torch.mean(torch.abs(enhanced - original))

        # 曝光控制损失
        mean_val = torch.mean(enhanced, dim=(2, 3), keepdim=True)
        loss_exposure = torch.mean(torch.abs(mean_val - 0.6))

        # 颜色恒常性损失
        enhanced_avg = torch.mean(enhanced, dim=(2, 3))
        original_avg = torch.mean(original, dim=(2, 3))
        loss_color = torch.mean(torch.abs(enhanced_avg - original_avg))

        # 光照平滑损失
        tv_loss = torch.mean(torch.abs(enhanced[:, :, :, :-1] - enhanced[:, :, :, 1:])) + \
                  torch.mean(torch.abs(enhanced[:, :, :-1, :] - enhanced[:, :, 1:, :]))

        return 20 * loss_spatial + 10 * loss_exposure + 5 * loss_color + 200 * tv_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        enhanced = self.model(x)
        loss = self._loss_function(enhanced, x)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)


# ----------------------
# 3. 完整训练流程
# ----------------------
def full_training():
    # 阶段一：训练图像增强器
    print("Training Zero-DCE Enhancer...")
    enhancer = ZeroDCE()

    full_dataset = EyeDataset(
        'Training_Dataset.xlsx',
        'images/',
        transform=get_transforms('train')
    )

    # 数据集分割
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # 可适当调整批次大小batch_size 4-16
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32)

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        gradient_clip_val=0.5,  # 添加梯度裁剪
        devices=1,
        callbacks=[
            ModelCheckpoint(monitor='train_loss', mode='min')
        ]
    )
    trainer.fit(enhancer, train_loader, val_loader)
    torch.save(enhancer.model.state_dict(), 'zero_dce.pth')


# ----------------------
# 4. 执行与测试
# ----------------------
if __name__ == '__main__':
    # 完整训练流程
    full_training()