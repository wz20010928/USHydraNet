import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import SimpleITK as sitk
from torchvision import transforms
import torch.nn.functional as F
from torch import amp
import torchvision.transforms.functional as TF

# 修改路径，确保能找到 echo.py
import sys
sys.path.append('/home/gem/wz/心脏/混合/单一')

# 导入 Echo 类
try:
    from echo import Echo
    print("Echo class imported successfully from /home/gem/wz/心脏/混合/单一.")
except ImportError as e:
    print(f"Error importing Echo: {e}")

# 提升 GPU 性能的设置
torch.backends.cudnn.benchmark = True  # 对于固定输入尺寸非常有用

# 数据预处理
both_transform = transforms.Compose([
    transforms.Resize((112, 112)),  # 将图像尺寸调整为112x112
    transforms.ToTensor(),
])

# 自定义 CAMUS 数据集类
class CAMUS_4CH_Dataset(Dataset):
    def __init__(self, data_dir, split='train'):
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

        image_file = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkFloat32))
        mask_file = sitk.GetArrayFromImage(sitk.ReadImage(mask_path, sitk.sitkFloat32))

        mask_file[mask_file != 1] = 0
        mask_file[mask_file == 1] = 1.0

        if image_file.ndim == 2:
            image_file = np.expand_dims(image_file, axis=0)

        if image_file.shape[0] == 1:
            image_file = np.repeat(image_file, 3, axis=0)

        image_file = torch.from_numpy(image_file).float()
        mask_file = torch.from_numpy(np.expand_dims(mask_file, axis=0)).float()

        if image_file.shape[1:] != torch.Size([112, 112]):  # 修改为112x112
            image_file = TF.resize(image_file, [112, 112])
        if mask_file.shape[1:] != torch.Size([112, 112]):  # 修改为112x112
            mask_file = TF.resize(mask_file, [112, 112])

        image_file = image_file.view(3, 112, 112)  # 调整为112x112
        mask_file = mask_file.view(1, 112, 112)  # 调整为112x112

        return image_file, mask_file

# 自定义 Echo 数据集类
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

        if image.ndim == 4 and image.shape[1] > 1:
            image = image[:, 0, :, :]

        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 2:
            image = image[np.newaxis, ...]

        if mask_image.ndim == 3:
            mask_image = mask_image[0][np.newaxis, ...]

        image = torch.from_numpy(image).float()
        mask_image = torch.from_numpy(mask_image).float()

        if image.shape[1:] != torch.Size([112, 112]):  # 修改为112x112
            image = TF.resize(image, [112, 112])
        if mask_image.shape[1:] != torch.Size([112, 112]):  # 修改为112x112
            mask_image = TF.resize(mask_image, [112, 112])

        image = image.view(3, 112, 112)  # 调整为112x112
        mask_image = mask_image.view(1, 112, 112)  # 调整为112x112

        return image, mask_image

# 定义 CombinedDataset 将两个数据集组合
class CombinedDataset(Dataset):
    def __init__(self, camus_dataset, echo_dataset):
        self.camus_dataset = camus_dataset
        self.echo_dataset = echo_dataset
        self.total_len = len(camus_dataset) + len(echo_dataset)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < len(self.camus_dataset):
            return self.camus_dataset[idx]
        else:
            return self.echo_dataset[idx - len(self.camus_dataset)]

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
        
        self.encoder = nn.ModuleList([conv_block(3, 64), conv_block(64, 128), conv_block(128, 256), conv_block(256, 512)])
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])
        
        self.decoder = nn.ModuleList([conv_block(1024, 512), conv_block(512, 256), conv_block(256, 128), conv_block(128, 64)])
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
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
        
        x = self.final_conv(x)
        return x

# 使用混合精度训练以提高 GPU 效率
scaler = amp.GradScaler()  # 自动混合精度

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
    accuracy = (tp + tn + smooth) / (y_true_f.size(0) + smooth)
    return dice.item(), iou.item(), precision.item(), recall.item(), accuracy.item()

# 验证函数
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    iou_score = 0.0
    precision_score = 0.0
    recall_score = 0.0
    accuracy_score = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                dice, iou, precision, recall, accuracy = calculate_metrics(masks, outputs)
                dice_score += dice
                iou_score += iou
                precision_score += precision
                recall_score += recall
                accuracy_score += accuracy

    val_loss /= len(loader)
    metrics = {
        'loss': val_loss,
        'dice': dice_score / len(loader),
        'iou': iou_score / len(loader),
        'precision': precision_score / len(loader),
        'recall': recall_score / len(loader),
        'accuracy': accuracy_score / len(loader),
    }
    return metrics

# 训练函数
def train_model(model, train_loader, val_loader, test_loader, num_epochs=50, learning_rate=0.001, accumulation_steps=4):
    # 指定使用 GPU 3
    device = torch.device(f"cuda:3" if torch.cuda.is_available() else "cpu")
    
    # 直接将模型移动到 GPU 3，不使用多 GPU
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scaler = amp.GradScaler('cuda')
    
    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        dice_score = 0.0
        iou_score = 0.0
        precision_score = 0.0
        recall_score = 0.0
        accuracy_score = 0.0
        optimizer.zero_grad()

        for step, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulation_steps  # 梯度累积
            
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            dice, iou, precision, recall, accuracy = calculate_metrics(masks, outputs)
            dice_score += dice
            iou_score += iou
            precision_score += precision
            recall_score += recall
            accuracy_score += accuracy

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Dice: {dice_score/len(train_loader):.4f}, IoU: {iou_score/len(train_loader):.4f}, Precision: {precision_score/len(train_loader):.4f}, Recall: {recall_score/len(train_loader):.4f}, Accuracy: {accuracy_score/len(train_loader):.4f}")
        
        # 验证集评估
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

    # 测试集评估
    print("开始测试...")
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice: {test_metrics['dice']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")

# 数据集路径
DATA_DIR_CAMUS = "/home/gem/wz/心脏/CAMUS_public/CAMUS_public/database_nifti"
DATA_DIR_ECHO = "/home/gem/wz/心脏/EchoNet-Dynamic"

# 加载 EchoNet-Dynamic 数据集的完整数据
echo_data_all = Echo(
    root=DATA_DIR_ECHO,
    split="all",  # 加载所有数据
    target_type=["LargeFrame", "LargeTrace"],
    length=1,
    period=1,
    clips=1,
    pad=None,
    noise=None,
    mean=0,
    std=1,
)

# 将 Echo 数据集按照 6:2:2 划分为训练集、验证集和测试集
total_len = len(echo_data_all)
train_len = int(0.6 * total_len)
val_len = int(0.2 * total_len)
test_len = total_len - train_len - val_len

echo_train_data, echo_val_data, echo_test_data = random_split(echo_data_all, [train_len, val_len, test_len])

# 初始化 CAMUS 数据集
camus_train_data = CAMUS_4CH_Dataset(DATA_DIR_CAMUS, split='train')
camus_val_data = CAMUS_4CH_Dataset(DATA_DIR_CAMUS, split='val')
camus_test_data = CAMUS_4CH_Dataset(DATA_DIR_CAMUS, split='test')  # 定义 CAMUS 的测试集

# 包装为 EchoDataset
echo_dataset_train = EchoDataset(echo_train_data)
echo_dataset_val = EchoDataset(echo_val_data)
echo_dataset_test = EchoDataset(echo_test_data)  # 定义 Echo 测试集

combined_dataset_train = CombinedDataset(camus_train_data, echo_dataset_train)
combined_dataset_val = CombinedDataset(camus_val_data, echo_dataset_val)

# 创建 DataLoader
batch_size = 128  # 保持 batch size 不变
num_workers = 16  # 保持 num_workers 不变
pin_memory = True  # 加速数据从内存到 GPU 的传输

train_loader = DataLoader(combined_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(combined_dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(camus_test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)  # 使用 CAMUS 测试集

# 打印 CAMUS 和 Echo 数据集的训练集、验证集和测试集的样本数
print(f"CAMUS 训练集样本数: {len(camus_train_data)}")
print(f"CAMUS 验证集样本数: {len(camus_val_data)}")
print(f"CAMUS 测试集样本数: {len(camus_test_data)}")  # 打印 CAMUS 测试集的样本数
print(f"Echo 训练集样本数: {len(echo_dataset_train)}")
print(f"Echo 验证集样本数: {len(echo_dataset_val)}")
print(f"Echo 测试集样本数: {len(echo_dataset_test)}")  # 打印 Echo 测试集的样本数

# 打印组合后的训练集和验证集样本数
print(f"现在训练的训练集样本数: {len(train_loader.dataset)}")
print(f"现在训练的验证集样本数: {len(val_loader.dataset)}")
print(f"现在训练的测试集样本数: {len(test_loader.dataset)}")

# 创建模型并训练
model = UNet()
train_model(model, train_loader, val_loader, test_loader, num_epochs=100)
