import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import os
from matplotlib.colors import ListedColormap
from tqdm import tqdm  # 添加进度条支持

plt.close('all')


def load_raw_data(file_path, target_shape=(256, 256, 256)):
    """
    从RAW文件加载数据并调整到目标形状
    """
    # 读取原始数据并重塑
    rockdata = np.fromfile(file_path, dtype=np.uint8)

    # 计算原始形状
    original_shape = (333, 411, 411)
    rockdata = rockdata.reshape(original_shape)

    # 裁剪到目标形状
    return rockdata[:target_shape[0], :target_shape[1], :target_shape[2]]


def create_corrupted_sinogram(labeled_image, metal_mask, theta_range):
    """
    创建带金属伪影的sinogram
    """
    # 创建基础sinogram
    radon_combined = radon(labeled_image.astype(np.float64), theta_range, circle=False)

    # 创建金属部分的sinogram（用最大值2投影）
    radon_metal = radon(metal_mask.astype(np.float64) * 2, theta_range, circle=False)

    # 创建带伪影的sinogram（金属部分保持高值）
    corrupted_sinogram = radon_combined.copy()
    metal_indices = radon_metal > 0
    corrupted_sinogram[metal_indices] = radon_metal[metal_indices]

    return corrupted_sinogram, radon_combined


def reconstruct_and_quantize(sinogram, theta_range):
    """
    重建图像并进行量化
    """
    # 重建图像
    reconstructed = iradon(
        sinogram,
        theta_range,
        filter_name='shepp-logan',
        output_size=256,
        circle=False
    )

    return reconstructed


def visualize_results(original_slice, reconstructed_labels, clean_recon, slice_idx, save_dir=None):
    """
    可视化结果
    """
    custom_cmap = ListedColormap(['blue', 'green', 'red'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始切片
    axes[0].imshow(original_slice, cmap=custom_cmap, vmin=0, vmax=2)
    axes[0].set_title(f'Original Slice {slice_idx}')
    axes[0].set_axis_off()

    # 带伪影重建
    im1 = axes[1].imshow(reconstructed_labels, cmap='gray')
    axes[1].set_title(f'Reconstructed with Artifacts (Slice {slice_idx})')
    axes[1].set_axis_off()
    plt.colorbar(im1, ax=axes[1], ticks=[0, 1, 2])

    # 无伪影重建对比
    im2 = axes[2].imshow(clean_recon, cmap=custom_cmap, vmin=0, vmax=2)
    axes[2].set_title(f'Clean Reconstruction (Slice {slice_idx})')
    axes[2].set_axis_off()
    plt.colorbar(im2, ax=axes[2], ticks=[0, 1, 2])

    plt.tight_layout()

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
    theta_range = np.arange(0, 180, 1)  # 投影角度范围

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    print("正在加载数据...")
    rock_data = load_raw_data(image_path)
    print(f"数据加载完成，形状: {rock_data.shape}")
    print(f"原始数据的唯一值: {np.unique(rock_data)}")

    # 处理每个切片
    artifact_reconstructions = []

    # 使用tqdm添加进度条
    for i in tqdm(range(rock_data.shape[0]), desc="处理切片"):
        # 获取当前切片
        current_slice = rock_data[i].copy()

        # 创建各相掩码
        pore_mask = current_slice == 0  # 孔隙
        matrix_mask = current_slice == 1  # 基质
        metal_mask = current_slice == 2  # 金属

        # 创建带伪影的sinogram
        corrupted_sinogram, clean_sinogram = create_corrupted_sinogram(
            current_slice, metal_mask, theta_range
        )

        # 重建图像
        artifact_reconstruction = reconstruct_and_quantize(corrupted_sinogram, theta_range)
        clean_reconstruction = reconstruct_and_quantize(clean_sinogram, theta_range)

        # 存储重建结果
        artifact_reconstructions.append(artifact_reconstruction)

        # 可视化（每10个切片显示一次）
        if i % 10 == 0:
            visualize_results(
                current_slice,
                artifact_reconstruction,
                clean_reconstruction,
                i,
                save_dir
            )

    # 转换为3D数组
    artifact_array = np.array(artifact_reconstructions, dtype=np.float32)
    print(f"最终3D数组形状: {artifact_array.shape}")
    print(f"最终3D数组值范围: [{artifact_array.min():.3f}, {artifact_array.max():.3f}]")

    # 保存结果
    # 保存为3D RAW文件
    raw_filename = os.path.join(save_dir, 'artifact_reconstruction_3d.raw')
    with open(raw_filename, 'wb') as f:
        f.write(artifact_array.tobytes())
    print(f"3D RAW文件已保存: {raw_filename}")

    # 保存为numpy文件
    np.save(os.path.join(save_dir, 'artifact_reconstruction_3d.npy'), artifact_array)

    # 保存中间切片作为示例
    middle_slice = artifact_array[artifact_array.shape[0] // 2]
    # 归一化到0-255范围用于图像保存
    middle_slice_normalized = ((middle_slice - middle_slice.min()) /
                               (middle_slice.max() - middle_slice.min()) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, 'middle_slice_example.png'), middle_slice_normalized)

    print("处理完成！")


if __name__ == "__main__":
    main()