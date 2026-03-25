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

# 定义变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 数据集路径
root_dir = '/home/gem/wz/max3'
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')
test_dir = os.path.join(root_dir, 'test')

# 创建数据集
train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset = ImageFolder(root=val_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# 设置设备
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 4

# 加载ResNet18模型
model = models.resnet18(pretrained=False)

# 修改最后的全连接层
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 优化整个模型的权重

# 评估模型的函数
def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
    return avg_loss, precision, recall, f1, accuracy, auc

# 训练循环
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        torch.cuda.synchronize()  # 确保所有的GPU任务完成
    epoch_loss = running_loss / len(train_loader.dataset)
    end_time = time.time()
    print(f"Epoch {epoch + 1}/{num_epochs} completed in {end_time - start_time:.2f} seconds")

    # 在验证集上评估
    val_loss, val_precision, val_recall, val_f1, val_accuracy, val_auc = evaluate(model, val_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {epoch_loss:.4f}, '
          f'Val Loss: {val_loss:.4f}, '
          f'Val Precision: {val_precision:.4f}, '
          f'Val Recall: {val_recall:.4f}, '
          f'Val F1-score: {val_f1:.4f}, '
          f'Val Accuracy: {val_accuracy:.4f}, '
          f'Val AUC: {val_auc:.4f}')

# 在训练集和测试集上评估
train_loss, train_precision, train_recall, train_f1, train_accuracy, train_auc = evaluate(model, train_loader)
print(f'Train Loss: {train_loss:.4f}')
print(f'Train Precision: {train_precision:.4f}')
print(f'Train Recall: {train_recall:.4f}')
print(f'Train F1-score: {train_f1:.4f}')
print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Train AUC: {train_auc:.4f}')

# 在测试集上评估
test_loss, test_precision, test_recall, test_f1, test_accuracy, test_auc = evaluate(model, test_loader)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test F1-score: {test_f1:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test AUC: {test_auc:.4f}')
