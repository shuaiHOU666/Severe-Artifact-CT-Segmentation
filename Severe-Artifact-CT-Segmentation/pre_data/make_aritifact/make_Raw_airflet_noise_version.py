import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import os
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

plt.close('all')


def load_raw_data(file_path, target_shape=(256, 256, 256)):
    """
    从RAW文件加载数据并调整到目标形状
    """
    rockdata = np.fromfile(file_path, dtype=np.uint8)
    original_shape = (333, 411, 411)
    rockdata = rockdata.reshape(original_shape)
    return rockdata[:target_shape[0], :target_shape[1], :target_shape[2]]


def discrete_to_continuous(image, base_values=[60, 128, 200], noise_intensity=8):
    """
    将离散的三相数据转换为连续的灰度图像
    """
    continuous_image = np.zeros_like(image, dtype=np.float32)

    for i, base_val in enumerate(base_values):
        mask = image == i
        if np.any(mask):
            noise = np.random.normal(0, noise_intensity, size=np.sum(mask))
            continuous_image[mask] = np.clip(base_val + noise, 0, 255)

    return continuous_image.astype(np.uint8)


def create_corrupted_sinogram(continuous_image, original_labels, theta_range):
    """
    创建带金属伪影的sinogram - 修正版本
    """
    # 创建基础sinogram（保持原始值范围）
    radon_combined = radon(continuous_image.astype(np.float64), theta_range, circle=False)

    # 创建金属掩码投影（用于识别金属区域）
    metal_mask = (original_labels == 2).astype(np.float64)
    radon_metal = radon(metal_mask * 255, theta_range, circle=False)  # 金属区域用最大值

    # 创建带伪影的sinogram（在金属区域添加高强度信号）
    corrupted_sinogram = radon_combined.copy()

    # 找到金属投影区域（使用相对阈值）
    metal_threshold = np.max(radon_metal) * 0.3
    metal_indices = radon_metal > metal_threshold

    # 在金属区域添加伪影（增强信号但不过度）
    if np.any(metal_indices):
        max_original = np.max(radon_combined)
        enhancement_factor = 2.5  # 适度的增强因子

        # 在金属区域增强信号
        corrupted_sinogram[metal_indices] = np.clip(
            radon_combined[metal_indices] * enhancement_factor,
            0, max_original * 3  # 限制最大值
        )

    return corrupted_sinogram, radon_combined, metal_mask


def reconstruct_and_normalize(sinogram, theta_range, reference_range=None):
    """
    重建图像并进行合理的归一化
    """
    # 重建图像
    reconstructed = iradon(
        sinogram,
        theta_range,
        filter_name='ramp',  # 使用ramp滤波器减少伪影
        output_size=256,
        circle=False
    )

    # 如果提供了参考范围，使用相同的归一化参数
    if reference_range is not None:
        min_ref, max_ref = reference_range
        normalized = (reconstructed - min_ref) / (max_ref - min_ref) * 255
    else:
        # 使用百分位裁剪避免极端值影响
        p1, p99 = np.percentile(reconstructed, [1, 99])
        clipped = np.clip(reconstructed, p1, p99)
        normalized = (clipped - p1) / (p99 - p1) * 255

    return np.clip(normalized, 0, 255).astype(np.uint8), reconstructed


def visualize_comparison(original_discrete, continuous_image,
                         reconstructed_artifact, reconstructed_clean,
                         raw_artifact, raw_clean, slice_idx, save_dir=None):
    """
    可视化比较结果，包括原始重建值
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 原始离散图像
    custom_cmap = ListedColormap(['darkblue', 'green', 'red'])
    im0 = axes[0, 0].imshow(original_discrete, cmap=custom_cmap, vmin=0, vmax=2)
    axes[0, 0].set_title(f'Original Discrete (Slice {slice_idx})')
    axes[0, 0].set_axis_off()
    plt.colorbar(im0, ax=axes[0, 0], ticks=[0, 1, 2])

    # 连续灰度图像
    im1 = axes[0, 1].imshow(continuous_image, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Continuous Grayscale')
    axes[0, 1].set_axis_off()
    plt.colorbar(im1, ax=axes[0, 1])

    # 带伪影重建（归一化后）
    im2 = axes[0, 2].imshow(reconstructed_artifact, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title('With Metal Artifacts (Normalized)')
    axes[0, 2].set_axis_off()
    plt.colorbar(im2, ax=axes[0, 2])

    # 无伪影重建（归一化后）
    im3 = axes[1, 0].imshow(reconstructed_clean, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Clean Reconstruction (Normalized)')
    axes[1, 0].set_axis_off()
    plt.colorbar(im3, ax=axes[1, 0])

    # 带伪影重建（原始值）
    im4 = axes[1, 1].imshow(raw_artifact, cmap='viridis')
    axes[1, 1].set_title('Raw Artifact Reconstruction')
    axes[1, 1].set_axis_off()
    plt.colorbar(im4, ax=axes[1, 1])

    # 无伪影重建（原始值）
    im5 = axes[1, 2].imshow(raw_clean, cmap='viridis')
    axes[1, 2].set_title('Raw Clean Reconstruction')
    axes[1, 2].set_axis_off()
    plt.colorbar(im5, ax=axes[1, 2])

    # 添加值范围信息
    fig.text(0.1, 0.02, f'Artifact raw range: [{raw_artifact.min():.1f}, {raw_artifact.max():.1f}]', fontsize=10)
    fig.text(0.1, 0.01, f'Clean raw range: [{raw_clean.min():.1f}, {raw_clean.max():.1f}]', fontsize=10)

    plt.tight_layout()
    # plt.show()

    if save_dir:
        plt.savefig(os.path.join(save_dir, f'slice_{slice_idx:03d}_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def main():
    # 配置参数
    image_path = 'jinshu.411x411x333.raw'
    save_dir = './metal_artifacts/'
    theta_range = np.arange(0, 180, 1)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    print("正在加载数据...")
    rock_data = load_raw_data(image_path)
    print(f"数据形状: {rock_data.shape}, 唯一值: {np.unique(rock_data)}")

    # 处理每个切片
    artifact_reconstructions = []
    continuous_images = []
    raw_artifact_values = []
    raw_clean_values = []

    for i in tqdm(range(rock_data.shape[0]), desc="处理切片"):  # 先处理前50个测试
        original_slice = rock_data[i].copy()

        # 转换为连续灰度图像
        continuous_image = discrete_to_continuous(original_slice)

        # 创建sinogram
        corrupted_sinogram, clean_sinogram, metal_mask = create_corrupted_sinogram(
            continuous_image, original_slice, theta_range
        )

        # 首先重建无伪影图像并获取其值范围
        clean_normalized, raw_clean = reconstruct_and_normalize(clean_sinogram, theta_range)
        clean_range = (np.min(raw_clean), np.max(raw_clean))

        # 使用相同的归一化参数重建带伪影图像
        artifact_normalized, raw_artifact = reconstruct_and_normalize(
            corrupted_sinogram, theta_range, clean_range
        )

        # 存储结果
        artifact_reconstructions.append(artifact_normalized)
        continuous_images.append(continuous_image)
        raw_artifact_values.append(raw_artifact)
        raw_clean_values.append(raw_clean)

        # 可视化（每5个切片显示一次）
        if i % 5 == 0:
            print(f"\nSlice {i}:")
            print(f"Continuous range: [{continuous_image.min()}, {continuous_image.max()}]")
            print(f"Artifact raw range: [{raw_artifact.min():.1f}, {raw_artifact.max():.1f}]")
            print(f"Clean raw range: [{raw_clean.min():.1f}, {raw_clean.max():.1f}]")

            visualize_comparison(
                original_slice,
                continuous_image,
                artifact_normalized,
                clean_normalized,
                raw_artifact,
                raw_clean,
                i,
                save_dir
            )

    # 转换为3D数组
    artifact_array = np.array(artifact_reconstructions, dtype=np.uint8)
    continuous_array = np.array(continuous_images, dtype=np.uint8)

    print(f"\n最终结果:")
    print(f"带伪影数据 - 形状: {artifact_array.shape}, 范围: {artifact_array.min()}-{artifact_array.max()}")
    print(f"连续数据 - 形状: {continuous_array.shape}, 范围: {continuous_array.min()}-{continuous_array.max()}")
    # 保存为TIFF文件
    print("正在保存TIFF文件...")
    import tifffile as tiff
    # 保存带伪影的3D TIFF
    artifact_tiff_path = os.path.join(save_dir, 'artifact_reconstruction_3d.tiff')
    tiff.imwrite(artifact_tiff_path, artifact_array)
    print(f"带伪影3D TIFF文件已保存: {artifact_tiff_path}")

    # 保存连续图像的3D TIFF
    continuous_tiff_path = os.path.join(save_dir, 'continuous_grayscale_3d.tiff')
    tiff.imwrite(continuous_tiff_path, continuous_array)
    print(f"连续原始3D TIFF文件已保存: {continuous_tiff_path}")

    # 保存原始离散数据的3D TIFF（可选）
    original_tiff_path = os.path.join(save_dir, 'original_discrete_3d.tiff')
    tiff.imwrite(original_tiff_path, rock_data[:len(artifact_array)].astype(np.uint8))
    print(f"原始离散3D TIFF文件已保存: {original_tiff_path}")

    # 保存结果
    np.save(os.path.join(save_dir, 'artifact_reconstruction_3d.npy'), artifact_array)
    np.save(os.path.join(save_dir, 'continuous_grayscale_3d.npy'), continuous_array)

    # 保存RAW文件
    with open(os.path.join(save_dir, 'artifact_3d.raw'), 'wb') as f:
        f.write(artifact_array.tobytes())
    with open(os.path.join(save_dir, 'continuous_3d.raw'), 'wb') as f:
        f.write(continuous_array.tobytes())

    print("处理完成！")


if __name__ == "__main__":
    main()