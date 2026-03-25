import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import torch
from sklearn.preprocessing import LabelBinarizer
from torchvision.datasets import ImageFolder
import time

# =======================================
#         图像变换定义
# =======================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# =======================================
#    查找图像文件的辅助函数
# =======================================
def find_image_file(base_path):
    extensions = ['.png', '.jpg', '.jpeg']
    for ext in extensions:
        file_path = base_path + ext
        if os.path.exists(file_path):
            return file_path, None
    return None, None

# =======================================
#    自定义欧洲数据集 Dataset 类
# =======================================
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, images_dir, transform=None):
        # 分号分隔
        self.data_frame = pd.read_csv(csv_file, sep=',')
        self.images_dir = images_dir
        self.transform = transform
        self.data_frame.columns = self.data_frame.columns.str.strip()
        lb = LabelBinarizer()
        # one-hot 编码
        self.labels = lb.fit_transform(self.data_frame['Plane'])
        self.classes = lb.classes_

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        fname = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('Image_name')].strip()
        base_img = os.path.join(self.images_dir, fname)
        img_path, _ = find_image_file(base_img)
        if img_path is None:
            raise FileNotFoundError(f"图像不存在: {base_img}")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 返回 one-hot 浮动标签
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

# =======================================
#         加载训练集
# =======================================
root_dir = '/home/gem/wz/max3'
train_dir = os.path.join(root_dir, 'train')
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# =======================================
#   加载欧洲数据集并替换验证集
# =======================================
european_root_dir = '/home/gem/wz/Europe'
european_images_dir = os.path.join(european_root_dir, 'Images')
european_test_csv = os.path.join(european_root_dir, 'test.csv')

european_test_dataset = CustomDataset(
    csv_file=european_test_csv,
    root_dir=european_root_dir,
    images_dir=european_images_dir,
    transform=test_transform
)
european_test_loader = DataLoader(
    european_test_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=1
)

# 将 val_loader 指向 欧洲测试集
val_loader = european_test_loader

# =======================================
#         模型及训练设置
# =======================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 4
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 修改最后一层为 num_classes 个输出
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def evaluate(model, dataloader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            # 如果标签是 one-hot（二维），取 argmax 得到类别索引
            if labels.dim() == 2:
                labels_idx = labels.argmax(dim=1)
            else:
                labels_idx = labels
            # 计算损失
            loss = criterion(outputs, labels_idx.long())
            total_loss += loss.item() * inputs.size(0)
            # 预测
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels_idx.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
    return avg_loss, precision, recall, f1, accuracy, auc

# =======================================
#           训练与验证循环
# =======================================
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        torch.cuda.synchronize()
    epoch_loss = running_loss / len(train_loader.dataset)
    # 在“欧洲测试集”（val_loader）上评估
    val_loss, val_prec, val_rec, val_f1, val_acc, val_auc = evaluate(model, val_loader)
    print(
        f"Epoch {epoch+1}/{num_epochs} "
        f"Train Loss: {epoch_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}, "
        f"Val Prec: {val_prec:.4f}, "
        f"Val Rec: {val_rec:.4f}, "
        f"Val F1: {val_f1:.4f}, "
        f"Val Acc: {val_acc:.4f}, "
        f"Val AUC: {val_auc:.4f}"
    )

# =======================================
#         训练集和欧洲测试集评估
# =======================================
train_loss, train_prec, train_rec, train_f1, train_acc, train_auc = evaluate(model, train_loader)
print(f"Train — Loss: {train_loss:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")

test_loss, test_prec, test_rec, test_f1, test_acc, test_auc = evaluate(model, european_test_loader)
print(f"Europe Test — Loss: {test_loss:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
