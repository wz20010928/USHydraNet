import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 指定文件路径
nii_file = "/home/gem/wz/心脏/max/test/0X1A3D565B371DC573.nii.gz"

# 加载 NIfTI 文件
img = nib.load(nii_file)
data = img.get_fdata()

# 查看数据形状
print("数据维度：", data.shape)

# 选择一个切片索引（这里取中间层）
slice_index = data.shape[2] // 2

# 提取该层的图像数据
slice_data = data[:, :, slice_index]

# 显示该层图像
plt.figure(figsize=(6, 6))
plt.imshow(slice_data.T, cmap="gray", origin="lower")
plt.title(f"Slice {slice_index}")
plt.axis("off")
plt.show()
