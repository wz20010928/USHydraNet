import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import torch
import timm
from sklearn.preprocessing import LabelBinarizer

def find_image_file(base_path):
    extensions = ['.png', '.jpg', '.jpeg']
    tried_paths = []
    for ext in extensions:
        file_path = base_path + ext
        tried_paths.append(file_path)
        if os.path.exists(file_path):
            return file_path, tried_paths
    return None, tried_paths

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, images_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file, sep=';')
        self.root_dir = root_dir
        self.images_dir = images_dir
        self.transform = transform
        self.data_frame.columns = self.data_frame.columns.str.strip()

        lb = LabelBinarizer()
        self.labels = lb.fit_transform(self.data_frame['Plane'])
        self.classes = lb.classes_

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        base_img_name = os.path.join(self.images_dir,
                                     self.data_frame.iloc[idx, self.data_frame.columns.get_loc('Image_name')].strip())

        img_name, tried_paths = find_image_file(base_img_name)
        if img_name is None:
            raise FileNotFoundError(f"Image file not found for base name: {base_img_name}")

        image = Image.open(img_name).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# 定义变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# CSV文件和图像存储的目录
root_dir = '/root/autodl-fs/.sys/FETAL_PLANES_ZENODO/FETAL_PLANES_ZENODO'
images_dir = os.path.join(root_dir, 'Images')

# 创建数据集
train_dataset = CustomDataset(csv_file=os.path.join(root_dir, 'train.csv'), root_dir=root_dir, images_dir=images_dir, transform=transform)
val_dataset = CustomDataset(csv_file=os.path.join(root_dir, 'val.csv'), root_dir=root_dir, images_dir=images_dir, transform=transform)
test_dataset = CustomDataset(csv_file=os.path.join(root_dir, 'test.csv'), root_dir=root_dir, images_dir=images_dir, transform=transform)

# 比较CSV图像名称与实际文件以查找差异
csv_image_names = set(train_dataset.data_frame['Image_name'].str.strip())
actual_image_names = set([os.path.splitext(f)[0] for f in os.listdir(images_dir)])

missing_files = csv_image_names - actual_image_names
if missing_files:
    print(f"Missing files: {missing_files}")
else:
    print("All files are present.")

datasets = [train_dataset, val_dataset, test_dataset]
dataset_names = ['train', 'validation', 'test']
for name, dataset in zip(dataset_names, datasets):
    print(f"Checking images for {name} dataset...")
    for idx in range(len(dataset)):
        try:
            dataset[idx]
        except FileNotFoundError as e:
            print(e)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 4

# 加载ViT模型和本地预训练权重
model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.load_state_dict(torch.load('/root/autodl-fs/jx_vit_base_p16_224-80ecf9dd.pth'))

# 移除分类头
num_features = model.head.in_features
model.head = nn.Identity()  # 将分类头替换为Identity层

# 定义新的Decoder作为分类头
class Decoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建Decoder模型
decoder = Decoder(num_features, num_classes)
model = nn.Sequential(model, decoder)
model = model.to(device)

# 冻结ViT模型的权重
for param in model[0].parameters():
    param.requires_grad = False

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model[1].parameters(), lr=0.0005)  # 只优化Decoder的权重

# 评估模型的函数
def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = model[0](inputs)
            outputs = model[1](features)
            loss = criterion(outputs, torch.max(labels, 1)[1])  # 使用类索引进行损失计算
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(torch.max(labels, 1)[1].cpu().numpy())
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
best_val_loss = float('inf')
best_decoder_weights = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        features = model[0](inputs)
        outputs = model[1](features)
        loss = criterion(outputs, torch.max(labels, 1)[1])  # 使用类索引进行损失计算
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        torch.cuda.empty_cache()  # 清理GPU缓存
    epoch_loss = running_loss / len(train_loader.dataset)

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

    # 保存最好的Decoder权重
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_decoder_weights = model[1].state_dict()

# 保存最好的Decoder权重
torch.save(best_decoder_weights, '/root/autodl-fs/best_decoder_weights.pth')

# 在测试集上评估
test_loss, test_precision, test_recall, test_f1, test_accuracy, test_auc = evaluate(model, test_loader)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test F1-score: {test_f1:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test AUC: {test_auc:.4f}')
