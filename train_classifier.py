import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import pytorch_lightning as pl
from timm import create_model
from sklearn.metrics import roc_auc_score, f1_score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


# ----------------------
# 1. 数据准备与预处理
# ----------------------
class EyeDataset(Dataset):
    def __init__(self, excel_path, img_root, transform=None):
        self.df = pd.read_excel(excel_path)
        self.img_root = img_root
        self.transform = transform
        self._process_data()

    def _process_data(self):
        # 合并左右眼数据为独立样本
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

        # 加载增强后的图像（如果已生成）
        if os.path.exists(f'enhanced_{row["path"]}'):
            img_path = f'enhanced_{row["path"]}'

        image = Image.open(img_path).convert('RGB')
        labels = row[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels)


# ----------------------
# 2. 数据增强与加载
# ----------------------
def get_transforms():
    return T.Compose([
        T.Resize(512),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


full_dataset = EyeDataset(
    excel_path='Training_Dataset.xlsx',
    img_root='images/',
    transform=get_transforms()
)

# 分层划分数据集（确保类别均衡）
from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    stratify=full_dataset.data[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].values.argmax(axis=1)
)

train_ds = torch.utils.data.Subset(full_dataset, train_indices)
val_ds = torch.utils.data.Subset(full_dataset, val_indices)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=4, pin_memory=True)


# ----------------------
# 3. 分类器模型定义
# ----------------------
class DiseaseClassifier(pl.LightningModule):
    def __init__(self, num_classes=8, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = create_model('convnext_base', pretrained=True, num_classes=0)
        self.classifier = nn.Linear(1024, num_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(self.dropout(features))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)

        # 安全计算 AUC（跳过单一类别）
        auc_scores = []
        for i in range(y.shape[1]):
            if len(torch.unique(y[:, i])) > 1:
                auc = roc_auc_score(y[:, i].cpu().numpy(), probs[:, i].cpu().numpy())
                auc_scores.append(auc)

        avg_auc = np.mean(auc_scores) if auc_scores else 0.0
        self.log('val_auc', avg_auc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# ----------------------
# 4. 训练配置与执行
# ----------------------
checkpoint_callback = ModelCheckpoint(
    monitor='val_auc',
    mode='max',
    filename='best-{epoch}-{val_auc:.2f}',
    save_top_k=3
)

early_stop_callback = EarlyStopping(
    monitor='val_auc',
    patience=5,
    mode='max'
)

trainer = pl.Trainer(
    max_epochs=30,
    accelerator='gpu',
    devices=1,
    precision="16",  # 修改为 16-mixed
    callbacks=[checkpoint_callback, early_stop_callback],
    accumulate_grad_batches=2  # 梯度累积
)

if __name__ == '__main__':
    model = DiseaseClassifier()
    trainer.fit(model, train_loader, val_loader)