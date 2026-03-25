import SimpleITK as sitk
import matplotlib.pyplot as plt

# 设置文件路径
image_path = '/home/gem/wz/心脏/CAMUS_public/CAMUS_public/database_nifti/patient0001/patient0001_4CH_ES.nii.gz'
mask_path = '/home/gem/wz/心脏/CAMUS_public/CAMUS_public/database_nifti/patient0001/patient0001_4CH_ES_gt.nii.gz'

# 读取图像和掩膜
image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(mask_path)

# 将SimpleITK图像转换为numpy数组
image_array = sitk.GetArrayFromImage(image)
mask_array = sitk.GetArrayFromImage(mask)

# 检查图像的维度
print(f"Image shape: {image_array.shape}")
print(f"Mask shape: {mask_array.shape}")

# 处理二维和三维的情况
if image_array.ndim == 3:
    # 如果是三维图像，选择中间切片
    slice_idx = image_array.shape[0] // 2
    image_slice = image_array[slice_idx, :, :]
    mask_slice = mask_array[slice_idx, :, :]
elif image_array.ndim == 2:
    # 如果是二维图像，直接显示
    image_slice = image_array
    mask_slice = mask_array
else:
    raise ValueError("Unsupported image dimensions.")

# 显示图像和掩膜
plt.figure(figsize=(12, 6))

# 显示图像
plt.subplot(1, 2, 1)
plt.imshow(image_slice, cmap='gray')
plt.title('Image Slice')
plt.axis('off')

# 显示掩膜
plt.subplot(1, 2, 2)
plt.imshow(mask_slice, cmap='gray')
plt.title('Mask Slice')
plt.axis('off')

# 确保调用 plt.show() 来渲染图像
plt.show()
