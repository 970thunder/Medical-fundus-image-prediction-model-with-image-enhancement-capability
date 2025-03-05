import os
import torch
import pandas as pd
from openpyxl import load_workbook
from torchvision import transforms
from PIL import Image
import numpy as np
import pytorch_lightning as pl
from timm import create_model


# ----------------------
# 1. 定义模型类
# ----------------------
class DiseaseClassifier(pl.LightningModule):
    def __init__(self, num_classes=8, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = create_model('convnext_base', pretrained=True, num_classes=0)
        self.classifier = torch.nn.Linear(1024, num_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(self.dropout(features))


# ----------------------
# 2. 加载训练好的模型
# ----------------------
checkpoint_path = './lightning_logs/version_2/checkpoints/best-epoch=10-val_auc=0.83.ckpt'
model = DiseaseClassifier.load_from_checkpoint(checkpoint_path)
model.eval()


# ----------------------
# 3. 数据预处理函数
# ----------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"无法加载图像 {image_path}: {str(e)}")
        return None


# ----------------------
# 4. 预测函数
# ----------------------
def predict_image(image_tensor, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        return torch.sigmoid(logits).cpu().numpy()[0]


# ----------------------
# 5. Excel处理函数
# ----------------------
def process_excel(input_path):
    # 读取原始数据
    df = pd.read_excel(input_path, sheet_name='Sheet1')

    # 加载工作簿用于写入
    book = load_workbook(input_path)
    writer = pd.ExcelWriter(input_path, engine='openpyxl')
    writer.book = book

    # 准备结果DataFrame
    result_columns = [
        'Left-Fundus', 'Right-Fundus', 'Left-Diagnostic Keywords', 'Right-Diagnostic Keywords',
        'Left_Normal', 'Left_Diabetes', 'Left_Glaucoma', 'Left_Cataract',
        'Left_AMD', 'Left_Hypertension', 'Left_Myopia', 'Left_Others',
        'Right_Normal', 'Right_Diabetes', 'Right_Glaucoma', 'Right_Cataract',
        'Right_AMD', 'Right_Hypertension', 'Right_Myopia', 'Right_Others'
    ]
    result_df = pd.DataFrame(columns=result_columns)

    # 遍历每一行数据
    for idx, row in df.iterrows():
        # 基础信息
        base_info = {
            'Left-Fundus': row['Left-Fundus'],
            'Right-Fundus': row['Right-Fundus'],
            'Left-Diagnostic Keywords': row['Left-Diagnostic Keywords'],
            'Right-Diagnostic Keywords': row['Right-Diagnostic Keywords']
        }

        # 预测左眼
        left_probs = [0.0] * 8
        left_path = os.path.join('images', row['Left-Fundus'])
        if os.path.exists(left_path):
            left_tensor = preprocess_image(left_path)
            if left_tensor is not None:
                left_probs = predict_image(left_tensor, model)

        # 预测右眼
        right_probs = [0.0] * 8
        right_path = os.path.join('images', row['Right-Fundus'])
        if os.path.exists(right_path):
            right_tensor = preprocess_image(right_path)
            if right_tensor is not None:
                right_probs = predict_image(right_tensor, model)

        # 合并结果
        result_row = {
            **base_info,
            'Left_Normal': left_probs[0],
            'Left_Diabetes': left_probs[1],
            'Left_Glaucoma': left_probs[2],
            'Left_Cataract': left_probs[3],
            'Left_AMD': left_probs[4],
            'Left_Hypertension': left_probs[5],
            'Left_Myopia': left_probs[6],
            'Left_Others': left_probs[7],
            'Right_Normal': right_probs[0],
            'Right_Diabetes': right_probs[1],
            'Right_Glaucoma': right_probs[2],
            'Right_Cataract': right_probs[3],
            'Right_AMD': right_probs[4],
            'Right_Hypertension': right_probs[5],
            'Right_Myopia': right_probs[6],
            'Right_Others': right_probs[7]
        }

        result_df = result_df.append(result_row, ignore_index=True)

    # 写入结果
    result_df.to_excel(writer, sheet_name='模型测试结果', index=False)
    writer.save()
    print("预测结果已成功写入Excel文件！")


# ----------------------
# 6. 主程序
# ----------------------
if __name__ == '__main__':
    excel_path = 'Training_Dataset.xlsx'
    process_excel(excel_path)