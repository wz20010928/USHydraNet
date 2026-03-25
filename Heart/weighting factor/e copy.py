import os
import sys
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch import amp  # 混合精度训练
import torch.nn.functional as F


# 将 echo.py 文件所在目录添加到 Python 路径中
sys.path.append('/home/gem/wz/心脏/权重/echo.py')

# 从 echo.py 导入 Echo 类
from echo import Echo

# 自定义 Dataset 类
class EchoDataset(Dataset):
    def __init__(self, echo_data):
        self.echo_data = echo_data

    def __len__(self):
        return len(self.echo_data)

    def __getitem__(self, idx):
        image, mask = self.echo_data[idx]

        if isinstance(mask, tuple) and len(mask) == 2:
            large_trace = mask[0]
        else:
            raise ValueError("mask should be a tuple containing two elements.")

        mask_image = (large_trace > 0).astype(np.float32)

        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))  # [C, H, W]
        elif image.ndim == 2:
            image = image[np.newaxis, ...]

        if mask_image.ndim == 3:
            mask_image = mask_image[0][np.newaxis, ...]

        return torch.from_numpy(image).float(), torch.from_numpy(mask_image).float()

# 加载 EchoNet-Dynamic 数据集
data_path = '/home/gem/wz/心脏/EchoNet-Dynamic'

train_data = Echo(
    root=data_path,
    split="train",
    target_type=["LargeFrame", "LargeTrace"],
    length=1,
    period=1,
    clips=1,
    pad=None,
    noise=None,
    mean=0,
    std=1,
)

val_data = Echo(
    root=data_path,
    split="val",
    target_type=["LargeFrame", "LargeTrace"],
    length=1,
    period=1,
    clips=1,
    pad=None,
    noise=None,
    mean=0,
    std=1,
)

test_data = Echo(
    root=data_path,
    split="test",
    target_type=["LargeFrame", "LargeTrace"],
    length=1,
    period=1,
    clips=1,
    pad=None,
    noise=None,
    mean=0,
    std=1,
)

batch_size = 32
num_workers = 8

train_dataset = EchoDataset(train_data)
val_dataset = EchoDataset(val_data)
test_dataset = EchoDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)

# 定义 UNet 模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.encoder = nn.ModuleList([
            conv_block(3, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512)
        ])
        
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])
        
        self.decoder = nn.ModuleList([
            conv_block(1024, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64)
        ])
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)   #输入64， 输出1
                # 最后一层卷积，输出1通道
        # self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  # 拼接后的32 + 32通道，输出1通道

    def forward(self, x):
        enc_outputs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outputs.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        for i in range(4):
            x = self.upconv[i](x)
            x = torch.cat([x, enc_outputs[3 - i]], dim=1)
            x = self.decoder[i](x)
        #x = self.final_conv(x)
        
        return x

class SegDecoder(nn.Module):
    def __init__(self):
        super(SegDecoder, self).__init__()
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        output =  self.final_conv(x)
                    # 上采样，将输出调整回 224x224
      #  output = F.interpolate(output, size=(224, 224), mode='bilinear', align_corners=True)
        
        return self.sigmoid(output)
    
class CombinedModel(nn.Module):
    def __init__(self, unet_model, decoder_camus):
        super(CombinedModel, self).__init__()
        self.unet = unet_model
        self.decoder_camus = decoder_camus
    
    def forward(self, x):
        features = self.unet(x)
        out1 = self.decoder_camus(features)
        return out1
    
# 定义分割指标
def calculate_metrics(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > threshold).float()

    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    intersection = (y_true_f * y_pred_f).sum()
    dice = (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

    union = y_true_f.sum() + y_pred_f.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    tp = (y_true_f * y_pred_f).sum()
    fp = ((1 - y_true_f) * y_pred_f).sum()
    precision = (tp + smooth) / (tp + fp + smooth)

    fn = (y_true_f * (1 - y_pred_f)).sum()
    recall = (tp + smooth) / (tp + fn + smooth)

    tn = ((1 - y_true_f) * (1 - y_pred_f)).sum()
    specificity = (tn + smooth) / (tn + fp + smooth)

    accuracy = (tp + tn + smooth) / (y_true_f.size(0) + smooth)
    f1_score = 2 * precision * recall / (precision + recall + smooth)

    return dice.item(), iou.item(), precision.item(), recall.item(), specificity.item(), accuracy.item(), f1_score.item()

# 验证函数，用于计算各项指标
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    iou_score = 0.0
    precision_score = 0.0
    recall_score = 0.0
    accuracy_score = 0.0
    f1_score_total = 0.0
    best_loss = float("inf")
    with torch.no_grad():
        for images, masks in loader:
            images = images.squeeze(2).to(device, non_blocking=True)
            masks = masks.squeeze(2).to(device, non_blocking=True)
            
            with amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                dice, iou, precision, recall, _, accuracy, f1_score = calculate_metrics(masks, outputs)
                dice_score += dice
                iou_score += iou
                precision_score += precision
                recall_score += recall
                accuracy_score += accuracy
                f1_score_total += f1_score

                # 保存最佳模型权重
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "/home/gem/wz/心脏/权重/best_model_ECHO_public20.pth")
            
    val_loss /= len(loader)
    metrics = {
        'loss': val_loss,
        'dice': dice_score / len(loader),
        'iou': iou_score / len(loader),
        'precision': precision_score / len(loader),
        'recall': recall_score / len(loader),
        'accuracy': accuracy_score / len(loader),
        'f1_score': f1_score_total / len(loader)
    }
    return metrics

# 定义训练函数
def train_model(model, train_loader, val_loader, test_loader, num_epochs=50, learning_rate=0.001, accumulation_steps=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scaler = amp.GradScaler('cuda')
    
    best_dice = 0.0  # 初始化最佳dice系数
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        dice_score = 0.0
        iou_score = 0.0
        precision_score = 0.0
        recall_score = 0.0
        specificity_score = 0.0
        accuracy_score = 0.0
        optimizer.zero_grad()

        for step, (images, masks) in enumerate(train_loader):
            images = images.squeeze(2).to(device, non_blocking=True)
            masks = masks.squeeze(2).to(device, non_blocking=True)
            
            with amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulation_steps  # 梯度累积
            
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            dice, iou, precision, recall, specificity, accuracy, f1_score = calculate_metrics(masks, outputs)
            dice_score += dice
            iou_score += iou
            precision_score += precision
            recall_score += recall
            specificity_score += specificity
            accuracy_score += accuracy

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Dice: {dice_score/len(train_loader):.4f}, IoU: {iou_score/len(train_loader):.4f}")
        
        # 验证集评估
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")

        # 保存最佳模型权重
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(model.state_dict(), "best_model_EchoNet-Dynamic11.pth")
            print(f"Best model saved with Dice: {best_dice:.4f}")

    # 测试集评估
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice: {test_metrics['dice']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1-Score: {test_metrics['f1_score']:.4f}")

# 创建模型并训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

decoder_camus = SegDecoder().to(device)
# 初始化模型
unet = UNet().to(device)

combined_model = CombinedModel(unet, decoder_camus).to(device)

model = combined_model

train_model(model, train_loader, val_loader, test_loader, num_epochs=20)
