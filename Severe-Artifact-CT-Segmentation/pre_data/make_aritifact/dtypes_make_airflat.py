import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.signal import convolve2d
import os

plt.close('all')

# 加载岩石切片图像
image_path = 'jinshu.411x411x333.raw'
rockdata = np.fromfile(image_path, dtype=np.uint8).reshape((333, 411, 411))[:256, :256, :256]
rockImage = rockdata[122].copy()
labeledImage = rockImage

print("切片122的唯一值:", np.unique(labeledImage))

# 投影参数
theta1 = np.arange(0, 180, 1)


# 改进的矿物簇伪影生成函数
def add_realistic_mineral_artifacts(image, base_intensity=150, max_artifact_intensity=80):
    """生成更真实的矿物伪影，包含矿物间的干涉效应"""
    artifact_image = image.copy().astype(np.float32)
    height, width = image.shape

    # 找到高亮矿物区域（值为2）
    mineral_mask = image == 2
    if np.sum(mineral_mask) == 0:
        return artifact_image

    # 增强矿物本身的亮度
    artifact_image[mineral_mask] = base_intensity

    # 计算每个矿物点到最近矿物点的距离场
    distance_to_minerals = distance_transform_edt(~mineral_mask)

    # 创建矿物密度图（高斯模糊矿物位置）
    mineral_density = mineral_mask.astype(float)
    mineral_density = gaussian_filter(mineral_density, sigma=3)

    # 计算伪影强度场 - 考虑距离衰减和矿物密度
    max_distance = 35
    decay_field = np.zeros_like(artifact_image)

    # 距离衰减：离矿物越远，伪影强度越小
    distance_decay = np.clip(1 - (distance_to_minerals / max_distance), 0, 1)

    # 密度增强：矿物密集区域伪影更强
    density_enhancement = 1 + mineral_density * 2

    # 组合效应
    artifact_strength = distance_decay * density_enhancement

    # 添加基础伪影
    base_artifacts = artifact_strength * max_artifact_intensity
    artifact_image += base_artifacts

    # 添加干涉条纹（矿物间相互作用）
    def add_interference_pattern(img, mineral_positions, wavelength=8, amplitude=15):
        """添加矿物间的干涉条纹"""
        interference = np.zeros_like(img)
        y, x = np.indices(img.shape)

        for i, (cy, cx) in enumerate(mineral_positions):
            if i >= 5:  # 只处理前5个主要矿物，避免计算量过大
                break

            # 计算到该矿物的距离
            dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

            # 创建干涉条纹（类似波干涉）
            pattern = amplitude * np.sin(2 * np.pi * dist / wavelength)

            # 距离衰减
            pattern *= np.exp(-dist / 20)

            interference += pattern

        return interference

    # 获取主要矿物位置
    mineral_positions = np.argwhere(mineral_mask)
    if len(mineral_positions) > 0:
        interference = add_interference_pattern(artifact_image, mineral_positions)
        artifact_image += interference

    # 添加散射效应（类似康普顿散射）
    def add_scattering_effect(img, mineral_mask, scattering_intensity=25):
        """添加散射效应"""
        scattering = np.zeros_like(img)

        # 创建散射核（各向异性）
        scatter_kernel = np.array([
            [0.1, 0.2, 0.1],
            [0.2, 1.0, 0.2],
            [0.1, 0.2, 0.1]
        ])

        # 对矿物区域进行卷积，模拟散射
        mineral_enhanced = mineral_mask.astype(float) * 50
        scattering = convolve2d(mineral_enhanced, scatter_kernel, mode='same', boundary='symm')

        # 距离衰减
        distance = distance_transform_edt(~mineral_mask)
        scattering *= np.exp(-distance / 15)

        return scattering * scattering_intensity

    scattering = add_scattering_effect(artifact_image, mineral_mask)
    artifact_image += scattering

    # 添加边缘增强效应（矿物边界处的伪影增强）
    from skimage.filters import sobel
    mineral_edges = sobel(mineral_mask.astype(float))
    edge_enhancement = mineral_edges * 20
    artifact_image += edge_enhancement

    # 确保不改变原始的非矿物区域结构
    non_mineral_mask = image != 2
    # 只对非矿物区域添加适度的伪影，保持原有结构
    artifact_image[non_mineral_mask] = np.clip(artifact_image[non_mineral_mask],
                                               image[non_mineral_mask] * 0.8,
                                               image[non_mineral_mask] * 1.5)

    return artifact_image


# 不同强度的伪影变体
def add_strong_mineral_artifacts(image):
    """强伪影版本"""
    return add_realistic_mineral_artifacts(image, base_intensity=180, max_artifact_intensity=100)


def add_weak_mineral_artifacts(image):
    """弱伪影版本"""
    return add_realistic_mineral_artifacts(image, base_intensity=130, max_artifact_intensity=50)


def add_focused_mineral_artifacts(image):
    """聚焦伪影版本（主要在矿物密集区域）"""
    artifact_image = image.copy().astype(np.float32)
    mineral_mask = image == 2

    if np.sum(mineral_mask) == 0:
        return artifact_image

    # 只在矿物密集区域添加强伪影
    mineral_density = gaussian_filter(mineral_mask.astype(float), sigma=5)
    dense_regions = mineral_density > 0.3

    # 增强密集区域的伪影
    distance = distance_transform_edt(~mineral_mask)
    artifact_strength = np.clip(1 - (distance / 25), 0, 1)
    artifact_strength = artifact_strength * dense_regions * 80

    artifact_image[mineral_mask] = 160
    artifact_image += artifact_strength

    return artifact_image


# 创建各种伪影版本
artifact_images = {
    'Original': labeledImage.astype(np.float32),
    'Realistic Mineral Artifacts': add_realistic_mineral_artifacts(labeledImage),
    'Strong Artifacts': add_strong_mineral_artifacts(labeledImage),
    'Weak Artifacts': add_weak_mineral_artifacts(labeledImage),
    'Focused Artifacts': add_focused_mineral_artifacts(labeledImage)
}

# 显示伪影图像
plt.figure(figsize=(16, 12))
for i, (title, img) in enumerate(artifact_images.items()):
    plt.subplot(2, 3, i + 1)
    im = plt.imshow(img, cmap='hot', vmin=0, vmax=200)
    plt.title(title, fontsize=10)
    plt.colorbar(im)
    plt.axis('off')

plt.tight_layout()
plt.show()

# 显示伪影的细节对比
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 原始图像中的矿物分布
mineral_mask = labeledImage == 2
im1 = axes[0, 0].imshow(mineral_mask, cmap='gray')
axes[0, 0].set_title('Original Mineral Distribution')
axes[0, 0].axis('off')

# 伪影强度分布
realistic_artifacts = artifact_images['Realistic Mineral Artifacts']
artifact_strength = realistic_artifacts - labeledImage.astype(np.float32)
im2 = axes[0, 1].imshow(artifact_strength, cmap='inferno')
axes[0, 1].set_title('Artifact Strength Distribution')
axes[0, 1].axis('off')
plt.colorbar(im2, ax=axes[0, 1])

# 伪影与矿物的关系
overlay = np.zeros((labeledImage.shape[0], labeledImage.shape[1], 3))
overlay[:, :, 0] = mineral_mask.astype(float)  # 红色：矿物
overlay[:, :, 1] = np.clip(artifact_strength / 80, 0, 1)  # 绿色：伪影强度
im3 = axes[1, 0].imshow(overlay)
axes[1, 0].set_title('Minerals (Red) vs Artifacts (Green)')
axes[1, 0].axis('off')

# 伪影的剖面线
if np.any(mineral_mask):
    center_y, center_x = np.argwhere(mineral_mask)[len(mineral_mask) // 2]
    profile = realistic_artifacts[center_y, :]
    original_profile = labeledImage[center_y, :]

    axes[1, 1].plot(profile, 'r-', label='With Artifacts')
    axes[1, 1].plot(original_profile, 'b-', label='Original')
    axes[1, 1].set_title('Intensity Profile')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Intensity')

plt.tight_layout()
plt.show()

# 生成sinogram并重建
results = {}

for title, artifact_image in artifact_images.items():
    print(f"处理: {title}")

    # 生成sinogram
    sinogram = radon(artifact_image, theta1, circle=False)

    # 重建图像
    reconstruction = iradon(sinogram, theta1, filter_name='ramp', output_size=256, circle=False)

    results[title] = {
        'original': artifact_image,
        'sinogram': sinogram,
        'reconstruction': reconstruction
    }

# 显示重建结果对比
fig, axes = plt.subplots(len(results), 3, figsize=(15, 5 * len(results)))

for i, (title, result) in enumerate(results.items()):
    # 原始图像（带伪影）
    im1 = axes[i, 0].imshow(result['original'], cmap='hot', vmin=0, vmax=200)
    axes[i, 0].set_title(f'{title}', fontsize=9)
    axes[i, 0].axis('off')

    # Sinogram
    im2 = axes[i, 1].imshow(result['sinogram'], cmap='gray',
                            extent=(0, 179, result['sinogram'].shape[0], 0))
    axes[i, 1].set_title('Sinogram', fontsize=9)
    axes[i, 1].set_xlabel('θ (degrees)')
    if i == 0:
        axes[i, 1].set_ylabel("t'")

    # 重建结果
    im3 = axes[i, 2].imshow(result['reconstruction'], cmap='hot')
    axes[i, 2].set_title('Reconstruction', fontsize=9)
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()

# 保存结果
savepath = './realistic_mineral_artifacts/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

for title, result in results.items():
    # 保存图像
    cv2.imwrite(os.path.join(savepath, f'{title}_original.png'),
                np.clip(result['original'], 0, 255).astype(np.uint8))

    # 保存数据
    np.save(os.path.join(savepath, f'{title}_original.npy'), result['original'])
    np.save(os.path.join(savepath, f'{title}_sinogram.npy'), result['sinogram'])
    np.save(os.path.join(savepath, f'{title}_reconstruction.npy'), result['reconstruction'])

print("所有真实矿物伪影结果已保存！")