import cv2


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tifffile import tifffile

# === 参数设置 ===
data = np.load('./metal_artifacts/continuous_grayscale_3d.npy')

# 随机添加环形伪影参数
ring_center = None  # 默认为图像中心 (x, y)
ring_radii = [40, 45, 50]  # 多个环的半径
ring_width = 3  # 衰减范围
ring_intensity = [-23, -20, -17]  # 每个环的强度（可正可负）

# 确定要添加伪影的切片数量
num_slices_to_modify = min(10, data.shape[0])  # 最多修改10个切片或所有可用切片

# 随机选择要修改的切片索引
slice_indices = np.random.choice(data.shape[0], num_slices_to_modify, replace=False)

# === 处理每个选中的切片 ===
modified_data = data.copy().astype(np.float32)

for slice_idx in slice_indices:
    img = data[slice_idx]
    h, w = img.shape

    if ring_center is None:
        ring_center = (w // 2, h // 2)

    # === 创建空的伪影图层 ===
    ring_artifact = np.zeros_like(img, dtype=np.float32)

    # 计算距离矩阵
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - ring_center[0]) ** 2 + (y - ring_center[1]) ** 2)

    # === 叠加多个高斯渐变环 ===
    sigma = ring_width / 2  # 高斯标准差

    for r, inten in zip(ring_radii, ring_intensity):
        gaussian_ring = np.exp(-((dist - r) ** 2) / (2 * sigma ** 2))
        ring_artifact += gaussian_ring * inten

    # === 添加伪影到图像 ===
    modified_data[slice_idx] = np.clip(img.astype(np.float32) + ring_artifact, 0, 255)

# 转换回原始数据类型
if data.dtype == np.uint8:
    modified_data = modified_data.astype(np.uint8)
elif data.dtype == np.uint16:
    modified_data = modified_data.astype(np.uint16)

# === 保存修改后的数据 ===
np.save('./metal_artifacts/continuous_grayscale_3d_with_rings.npy', modified_data)

tiff_output_path = './metal_artifacts/continuous_grayscale_3d_with_rings.tiff'
tifffile.imwrite(tiff_output_path, modified_data)

# === 显示示例切片 ===
plt.figure(figsize=(15, 5))

# 显示原始切片
plt.subplot(1, 3, 1)
plt.title("Original Slice")
plt.imshow(data[slice_indices[0]], cmap="gray")
plt.axis("off")

# 显示修改后的切片
plt.subplot(1, 3, 2)
plt.title("With Ring Artifacts")
plt.imshow(modified_data[slice_indices[0]], cmap="gray")
plt.axis("off")

# 显示差异
plt.subplot(1, 3, 3)
plt.title("Difference")
plt.imshow(modified_data[slice_indices[0]] - data[slice_indices[0]], cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"已成功在 {num_slices_to_modify} 个切片上添加环形伪影")
print(f"修改后的数据已保存为: './metal_artifacts/continuous_grayscale_3d_with_rings.npy'")