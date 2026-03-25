import torch

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np
from monai.metrics import DiceMetric
from sklearn.metrics import jaccard_score as jsc
from monai.transforms import AsDiscrete
from torch import amp 
import torch.cuda.amp as amp
import os
import torchvision.utils as vutils
#---------------------数据集---------------
sys.path.append('/home/gem/wz/心脏/权重')

from echo import Echo

# 自定义 CAMUS 数据集类
class CAMUS_4CH_Dataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.patients_list = sorted([x for x in os.listdir(data_dir) if (os.path.isdir(os.path.join(data_dir, x)) and x.startswith('patient'))])
        if split == 'train':
            self.patients_list = self.patients_list[:400]
        elif split == 'test':
            self.patients_list = self.patients_list[450:]

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

        if image_file.shape[1:] != torch.Size([112, 112]):
            image_file = TF.resize(image_file, [112, 112])
        if mask_file.shape[1:] != torch.Size([112, 112]):
            mask_file = TF.resize(mask_file, [112, 112])

        image_file = image_file.view(3, 112, 112)
        mask_file = mask_file.view(1, 112, 112)
       # print("CAMUS", image_file.shape) [3,112,112]  mask:[1,112,112]
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

        if image.shape[1:] != torch.Size([112, 112]):
            image = TF.resize(image, [112, 112])
        if mask_image.shape[1:] != torch.Size([112, 112]):
            mask_image = TF.resize(mask_image, [112, 112])

        image = image.view(3, 112, 112)
        mask_image = mask_image.view(1, 112, 112)

        return image, mask_image
#----------------------------------------------------------------------------------

#--------------数据集加载--------------------------------------------------------
# 初始化数据集
batch_size = 4 

DATA_DIR_CAMUS = "/home/gem/wz/心脏/CAMUS_public/CAMUS_public/database_nifti"
DATA_DIR_ECHO = "/home/gem/wz/心脏/EchoNet-Dynamic"
echo_data = Echo(root=DATA_DIR_ECHO, split="test", target_type=["LargeFrame", "LargeTrace"], length=1, period=1, clips=1)
camus_train_data = CAMUS_4CH_Dataset(DATA_DIR_CAMUS, split='train')
camus_test_data = CAMUS_4CH_Dataset(DATA_DIR_CAMUS, split='test')

echo_dataset = EchoDataset(echo_data)
# 按照6:2:2的比例划分 Echo 数据集的训练集、验证集和测试集
total_echo_size = len(echo_dataset)
train_size = int(0.6 * total_echo_size)
val_size = int(0.2 * total_echo_size)
test_size = total_echo_size - train_size - val_size
echo_train_data, echo_val_data, echo_test_data = random_split(echo_dataset, [train_size, val_size, test_size])


train_loader = DataLoader(echo_train_data, batch_size=batch_size, shuffle=True,)
val_loader = DataLoader(echo_val_data, batch_size=batch_size, shuffle=False, )
test_loader = DataLoader(echo_test_data, batch_size=batch_size, shuffle=False,)

camus_train_loader = DataLoader(camus_train_data,  batch_size=batch_size, shuffle=True)
camus_val_loader = DataLoader(camus_test_data, batch_size=batch_size, shuffle=False, )
camus_test_loader = DataLoader(camus_test_data, batch_size=batch_size, shuffle=False, num_workers=1)
echo_test_loader = DataLoader(echo_test_data, batch_size=batch_size, shuffle=False, num_workers=1)

# # 迭代 DataLoader 并打印每个批次的形状
# for images, masks in echo_test_loader:
#     print(f"图像批次形状: {images.shape}")  #torch.Size([bs, 3, 112, 112])
#     print(f"掩码批次形状: {masks.shape}")   #torch.Size([bs, 1, 112, 112])
#     break  
#-----------------------------------------------------------------------------------

#------------模型代码--------------------------------------------------------

# class DoubleDecoderUNet(nn.Module):
#     def __init__(self, num_classes1=1, num_classes2=1):
#         super(DoubleDecoderUNet, self).__init__()

#         def conv_block(in_channels, out_channels):
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True)
#             )

#         self.encoder = nn.ModuleList([
#             conv_block(3, 64),
#             conv_block(64, 128),
#             conv_block(128, 256),
#             conv_block(256, 512)
#         ])

#         self.pool = nn.MaxPool2d(2, 2)
#         self.bottleneck = conv_block(512, 1024)

#         self.upconv = nn.ModuleList([
#             nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
#             nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         ])

#         self.decoder1 = nn.ModuleList([
#             conv_block(1024, 512),
#             conv_block(512, 256),
#             conv_block(256, 128),
#             conv_block(128, 64)
#         ])

#         self.decoder2 = nn.ModuleList([
#             conv_block(1024, 512),
#             conv_block(512, 256),
#             conv_block(256, 128),
#             conv_block(128, 64)
#         ])

#         self.out1 = nn.Conv2d(64, num_classes1, kernel_size=1)
#         self.out2 = nn.Conv2d(64, num_classes2, kernel_size=1)


#     def forward(self, x):
#         enc_outputs = []
#         for enc in self.encoder:
#             x = enc(x)
#             enc_outputs.append(x)
#             x = self.pool(x)

#         x = self.bottleneck(x)

#         x1 = x
#         for i in range(4):
#             x1 = self.upconv[i](x1)
#             x1 = torch.cat([x1, enc_outputs[3 - i]], dim=1)
#             x1 = self.decoder1[i](x1)
#         out1 = self.out1(x1)


#         x2 = x
#         for i in range(4):
#             x2 = self.upconv[i](x2)
#             x2 = torch.cat([x2, enc_outputs[3 - i]], dim=1)
#             x2 = self.decoder2[i](x2)
#         out2 = self.out2(x2)

#         return out1, out2


#-------------------------------------------------------------------------------
# if __name__ == "__main__":
#     model = DoubleDecoderUNet()
#     input_tensor = torch.randn(2, 3, 112, 112)
#     output1, output2 = model(input_tensor)
#     print(output1.shape, output2.shape)


#-----------------------------------------------------------
#------------模型代码gai--------------------------------------------------------
class DoubleDecoderUNetInference(nn.Module):
    def __init__(self, num_classes1=1, num_classes2=1, mean_threshold=127.5, std_threshold=50.0):
        super(DoubleDecoderUNetInference, self).__init__()
        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold

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

        self.decoder1 = nn.ModuleList([
            conv_block(1024, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64)
        ])

        self.decoder2 = nn.ModuleList([
            conv_block(1024, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64)
        ])

        self.out1 = nn.Conv2d(64, num_classes1, kernel_size=1)
        self.out2 = nn.Conv2d(64, num_classes2, kernel_size=1)

    def forward(self, x):
        enc_outputs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        mean = x.mean()
        std = x.std()

        if mean > self.mean_threshold and std > self.std_threshold:
            x = x.clone() # Avoid inplace modification issues
            for i in range(4):
                x = self.upconv[i](x)
                x = torch.cat([x, enc_outputs[3 - i]], dim=1)
                x = self.decoder1[i](x)
            out = self.out1(x)
        else:
            x = x.clone() # Avoid inplace modification issues
            for i in range(4):
                x = self.upconv[i](x)
                x = torch.cat([x, enc_outputs[3 - i]], dim=1)
                x = self.decoder2[i](x)
            out = self.out2(x)

        return out

#-------------------------------------------------------    
# if __name__ == "__main__":
#     model = DoubleDecoderUNetInference(num_classes1=1, num_classes2=1, mean_threshold=100, std_threshold=60)
#     image = torch.randn(1, 3, 112, 112)  # 示例输入图像
#     output = model(image) # output 将是根据均值和方差选择的解码器的输出

#     print(output.shape)
#------开始训练--------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算分割评价指标
def dice_coefficient(preds, labels, smooth=1e-6):
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def calculate_iou(y_true, y_pred):
    y_true = y_true.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()
    return jsc(y_true, y_pred)

def iou_score(preds, labels, smooth=1e-6):
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def calculate_iou(y_true, y_pred):
    y_true = y_true.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()
    return jsc(y_true, y_pred)


#---------------------------------------------------------------
# 计算训练集的均值和方差
def compute_mean_std(loader, dataset_name):
    model.eval()
    original_features_list = []
    
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            original_features_list.append(inputs.cpu().numpy())
    
    original_features = np.concatenate(original_features_list, axis=0)

    # 计算每个通道的均值和方差
    original_mean = np.mean(original_features, axis=(0, 2, 3))  # 计算通道维度的均值
    original_std = np.std(original_features, axis=(0, 2, 3))    # 计算通道维度的方差
    
    print(f"{dataset_name} Dataset mean: {original_mean}, std: {original_std}")
    return original_mean, original_std

# 计算训练集均值和方差
# camus_mean, camus_std = compute_mean_std(DataLoader(camus_train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory), "CAMUS")
# echo_mean, echo_std = compute_mean_std(DataLoader(echo_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory), "Echo")
#------------------------------------------------------------------

def train_double_decoder(model, camus_test_loader, echo_test_loader, epochs=100, lr=1e-4, device="cuda", save_path="/home/gem/wz/心脏/推导/1224best_model.pth"):
    """
    Trains the double decoder UNet model and saves the best weights based on Dice and IoU scores.

    Args:
        model: The DoubleDecoderUNetInference model.
        camus_test_loader: DataLoader for the CAMUS test dataset.
        echo_test_loader: DataLoader for the Echo test dataset.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Device to train on ('cuda' or 'cpu').
        save_path: Path to save the best model weights.
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Example loss, adjust if needed
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    best_metric = -1
    post_trans = AsDiscrete(threshold=0.5) # post processing for calculating IoU

    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(camus_test_loader, desc=f"Epoch {epoch + 1}/{epochs} - CAMUS"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_dice = 0
            val_iou = 0
            num_val_batches = 0  # Counter for validation batches
            for batch in tqdm(echo_test_loader, desc=f"Validation - Echo"):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Dice calculation
                dice_metric(y_pred=outputs, y=targets)
                val_dice += dice_metric.aggregate().item()
                dice_metric.reset()

                # IoU calculation (requires post-processing)
                outputs = post_trans(outputs)
                val_iou += calculate_iou(targets, outputs)
                num_val_batches += 1

            val_dice /= num_val_batches
            val_iou /= num_val_batches
            metric = (val_dice + val_iou) / 2

            if metric > best_metric:
                best_metric = metric
                print(f"Saving best model at epoch {epoch + 1} with Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
                torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(camus_test_loader):.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")



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
def validate(model, loader, criterion, device, save_dir):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    iou_score = 0.0

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        val_bar = tqdm(loader, desc="Validation", unit="batch")

        for idx, (images, masks) in enumerate(val_bar):
            images = images.squeeze(2).to(device, non_blocking=True)
            masks = masks.squeeze(2).to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            dice, iou, _, _, _, _, _ = calculate_metrics(masks, outputs)
            dice_score += dice
            iou_score += iou

            # 保存原图、预测mask和ground truth
            save_path = os.path.join(save_dir, f"image_{idx}")
            os.makedirs(save_path, exist_ok=True)

            # 保存原图
            vutils.save_image(images, os.path.join(save_path, "original.png"), normalize=True)

            # 保存ground truth
            vutils.save_image(masks, os.path.join(save_path, "ground_truth.png"), normalize=True)

            # 保存预测mask
            vutils.save_image(outputs, os.path.join(save_path, "prediction.png"), normalize=True)

            val_bar.set_postfix(loss=f"{val_loss / (val_bar.n + 1):.4f}", 
                                dice=f"{dice_score / (val_bar.n + 1):.4f}", 
                                iou=f"{iou_score / (val_bar.n + 1):.4f}")

    val_loss /= len(loader)

    metrics = {
        'loss': val_loss,
        'dice': dice_score / len(loader),
        'iou': iou_score / len(loader),
    }
    print("OK")
    print(metrics)
    return metrics

# 定义训练函数
def train_model(model, train_loader, val_loader, test_loader, num_epochs=80, learning_rate=0.001, accumulation_steps=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    
    
    best_dice = 0.0  # 初始化最佳dice系数
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        dice_score = 0.0
        iou_score = 0.0
       
        optimizer.zero_grad()

         # Wrap the training data loader with tqdm for a progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for step, (images, masks) in enumerate(train_bar):
            images = images.squeeze(2).to(device, non_blocking=True)
            masks = masks.squeeze(2).to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks) / accumulation_steps

            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            dice, iou, _, _, _, _, _ = calculate_metrics(masks, outputs) # Underscores to ignore unused return values
            dice_score += dice
            iou_score += iou

            # Update the progress bar with current metrics
            train_bar.set_postfix(loss=f"{running_loss / (step + 1):.4f}", dice=f"{dice_score / (step + 1):.4f}", iou=f"{iou_score / (step + 1):.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Dice: {dice_score/len(train_loader):.4f}, IoU: {iou_score/len(train_loader):.4f}")

        # val_metrics = validate(model, val_loader, criterion, device)
        # print(f"Validation Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")

        # if val_metrics['dice'] > best_dice:
        #     best_dice = val_metrics['dice']
        #     torch.save(model.state_dict(), "best_model_EchoNet-Dynamic1225-cam.pth")
        #     print(f"Best model saved with Dice: {best_dice:.4f}")
            
            
    # # 测试集评估
    # test_metrics = validate(model, test_loader, criterion, device)
    # print(f"Test Loss: {test_metrics['loss']:.4f}, IoU: {test_metrics['iou']:.4f}, Dice: {test_metrics['dice']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1-Score: {test_metrics['f1_score']:.4f}")



if __name__ == "__main__":
    
    # 创建模型并训练
    model = DoubleDecoderUNetInference(mean_threshold=100, std_threshold=60)
    # train_model(model, camus_train_loader, camus_val_loader, camus_test_loader, num_epochs=80)
    train_model(model, camus_train_loader, camus_val_loader, camus_test_loader, num_epochs=80)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    validate(model, camus_test_loader, criterion, device, save_dir='/home/gem/wz/222')
    # image = torch.randn(1, 3, 112, 112)  # 示例输入图像
    # output = model(image) # output 将是根据均值和方差选择的解码器的输出
    # print(output.shape)
    # train_double_decoder(model, camus_test_loader, echo_test_loader, epochs=100, lr=1e-4)
    
