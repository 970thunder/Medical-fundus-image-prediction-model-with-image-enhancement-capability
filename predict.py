import os
import torch
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
checkpoint_path = './lightning_logs/version_2/checkpoints/best-epoch=10-val_auc=0.83.ckpt'  # 替换为实际的检查点路径
model = DiseaseClassifier.load_from_checkpoint(checkpoint_path)
model.eval()  # 设置为评估模式


# ----------------------
# 3. 数据预处理
# ----------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image


# ----------------------
# 4. 预测单张图片
# ----------------------
def predict_image(image_path, model):
    # 预处理图像
    image_tensor = preprocess_image(image_path)

    # 使用 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    # 预测
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # 转换为概率

    return probs


# ----------------------
# 5. 输出预测结果
# ----------------------
def print_prediction(probs):
    labels = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']
    print("预测结果：")
    for label, prob in zip(labels, probs):
        print(f"- {label}: {prob * 100:.2f}%")


# ----------------------
# 6. 测试单张图片
# ----------------------
if __name__ == '__main__':
    # 替换为要测试的图片路径
    test_image_path = 'TestImage/q5.JPG'
    # 预测
    probs = predict_image(test_image_path, model)
    # 输出结果
    print_prediction(probs)