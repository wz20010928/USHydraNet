import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import SimpleITK as sitk
import torch.nn.functional as F

# 图像和掩膜变换
both_transform = A.Compose(
    [A.Resize(width=224, height=224)],
    additional_targets={"image0": "image"},  # "image0" 用于掩膜
)

# 输入图像变换
transform_only_input = A.Compose(
    [
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=1.0),
        ToTensorV2(),
    ]
)

# 掩膜变换
transform_only_mask = A.Compose(
    [
        ToTensorV2(),
    ]
)

# 自定义数据集类
class CAMUS_4CH_Dataset(Dataset):
    def __init__(self, data_dir, split='train', trans=None):
        self.trans = trans
        self.data_dir = data_dir
        self.patients_list = sorted([x for x in os.listdir(data_dir) if (os.path.isdir(os.path.join(data_dir, x)) and x.startswith('patient'))])
        if split == 'train':
            self.patients_list = self.patients_list[:400]  # 训练集
        elif split == 'val':
            self.patients_list = self.patients_list[400:450]  # 验证集
        elif split == 'test':
            self.patients_list = self.patients_list[450:]  # 测试集

    def __len__(self):
        return len(self.patients_list)

    def __getitem__(self, index):
        patient_unique_id = str(self.patients_list[index])
        patient_dir = os.path.join(self.data_dir, patient_unique_id)
        image_path = os.path.join(patient_dir, patient_unique_id + "_4CH_ES.nii.gz")
        mask_path = os.path.join(patient_dir, patient_unique_id + "_4CH_ES_gt.nii.gz")
        
        # 使用SimpleITK加载图像和掩膜
        image_file = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkFloat32))
        mask_file = sitk.GetArrayFromImage(sitk.ReadImage(mask_path, sitk.sitkFloat32))

        # 掩膜预处理
        mask_file[mask_file != 1] = 0
        mask_file[mask_file == 1] = 1.0
        
        # 数据增强
        augmentations = both_transform(image=image_file, image0=mask_file)
        image_file, extended_mask = augmentations["image"], augmentations["image0"]

        # 转换为Tensor
        image_file = transform_only_input(image=image_file)["image"]
        extended_mask = transform_only_mask(image=extended_mask)["image"]

        if self.trans:
            return self.trans(image_file), extended_mask
        else:
            return image_file, extended_mask

# UNet 模型定义
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 编码器部分
        self.encoder1 = self.conv_block(1, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.encoder4 = self.conv_block(128, 256)

        # 瓶颈层
        self.bottleneck = self.conv_block(256, 512)

        # 解码器部分
        self.upconv4 = self.upconv_block(512, 256)  # 输入瓶颈层(512通道)，输出256通道
        self.upconv3 = self.upconv_block(512, 128)  # 拼接后的256 + 256通道，输出128通道
        self.upconv2 = self.upconv_block(256, 64)   # 拼接后的128 + 128通道，输出64通道
        self.upconv1 = self.upconv_block(128, 32)   # 拼接后的64 + 64通道，输出32通道

        # 最后一层卷积，输出1通道
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  # 拼接后的32 + 32通道，输出1通道
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 编码路径
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # 瓶颈层
        b = self.bottleneck(e4)

        # 解码路径，逐步拼接
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)  # 拼接，d4(256) + e4(256) = 512
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)  # 拼接，d3(128) + e3(128) = 256
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)  # 拼接，d2(64) + e2(64) = 128
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)  # 拼接，d1(32) + e1(32) = 64

        # 最后一层卷积，输出分割结果
        output = self.final_conv(d1)
        
        # 上采样，将输出调整回 224x224
        output = F.interpolate(output, size=(224, 224), mode='bilinear', align_corners=True)
        
        return self.sigmoid(output)

# 计算IoU
def iou_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1.0) / (union + 1.0)

# 计算Dice系数
def dice_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + 1.0) / (pred.sum() + target.sum() + 1.0)

# 计算准确率
def accuracy(pred, target):
    pred = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return correct / total

# 计算精确率
def precision(pred, target):
    pred = (pred > 0.5).float()
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    return (true_positive + 1.0) / (predicted_positive + 1.0)

# 计算召回率
def recall(pred, target):
    pred = (pred > 0.5).float()
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    return (true_positive + 1.0) / (actual_positive + 1.0)

# 计算F1分数
def f1_score(pred, target):
    p = precision(pred, target)
    r = recall(pred, target)
    return 2 * (p * r) / (p + r)

# 训练和验证函数
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_iou, total_dice, total_acc, total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_iou += iou_score(outputs, masks)
        total_dice += dice_score(outputs, masks)
        total_acc += accuracy(outputs, masks)
        total_precision += precision(outputs, masks)
        total_recall += recall(outputs, masks)
        total_f1 += f1_score(outputs, masks)
    metrics = {
        'loss': running_loss / len(loader),
        'iou': total_iou / len(loader),
        'dice': total_dice / len(loader),
        'accuracy': total_acc / len(loader),
        'precision': total_precision / len(loader),
        'recall': total_recall / len(loader),
        'f1_score': total_f1 / len(loader)
    }
    return metrics

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_iou, total_dice, total_acc, total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    best_loss = float("inf")
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            total_iou += iou_score(outputs, masks)
            total_dice += dice_score(outputs, masks)
            total_acc += accuracy(outputs, masks)
            total_precision += precision(outputs, masks)
            total_recall += recall(outputs, masks)
            total_f1 += f1_score(outputs, masks)

        # 保存最佳模型权重
        if running_loss < best_loss:
            best_loss = running_loss
            torch.save(model.state_dict(), "/home/gem/wz/心脏/权重/best_model_CAMUS_public.pth")

    metrics = {
        'loss': running_loss / len(loader),
        'iou': total_iou / len(loader),
        'dice': total_dice / len(loader),
        'accuracy': total_acc / len(loader),
        'precision': total_precision / len(loader),
        'recall': total_recall / len(loader),
        'f1_score': total_f1 / len(loader)
    }
    return metrics

# 设置路径和DataLoader
DATA_DIR = "/home/gem/wz/心脏/CAMUS_public/CAMUS_public/database_nifti"
train_data = CAMUS_4CH_Dataset(DATA_DIR, split='train')
val_data = CAMUS_4CH_Dataset(DATA_DIR, split='val')
test_data = CAMUS_4CH_Dataset(DATA_DIR, split='test')

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# 初始化模型、损失函数、优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和验证循环
num_epochs = 5
for epoch in range(num_epochs):
    train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_metrics = validate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, F1-Score: {train_metrics['f1_score']:.4f}")
    print(f"Val Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1-Score: {val_metrics['f1_score']:.4f}")

# 测试集评估
test_metrics = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_metrics['loss']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice: {test_metrics['dice']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1-Score: {test_metrics['f1_score']:.4f}")


