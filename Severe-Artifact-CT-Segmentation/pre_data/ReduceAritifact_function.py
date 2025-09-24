import numpy as np
from skimage.transform import radon, iradon
import cv2


def remove_metal_artifacts(image_array):
    """
    使用Radon变换去除金属伪影
    输入: 0-255范围的归一化图像数组
    输出: 去除伪影后的图像数组(0-255范围)
    """
    # 验证输入范围
    if np.min(image_array) < 0 or np.max(image_array) > 255:
        raise ValueError("输入数组必须在0-255范围内")

    # 使用最佳参数
    THRESHOLD = 90
    FILTER = 'ramp'

    # 1. 阈值分割 - 创建金属掩膜
    metal_mask = np.zeros_like(image_array)
    non_metal_image = np.zeros_like(image_array)
    metal_mask[image_array >= THRESHOLD] = 255
    non_metal_image[image_array < THRESHOLD] = image_array[image_array < THRESHOLD]

    # 2. Radon变换
    projection_angles = np.arange(0, 180, 1)
    radon_metal = radon(metal_mask, theta=projection_angles, circle=False)
    radon_non_metal = radon(non_metal_image, theta=projection_angles, circle=False)

    # 3. 边界检测和插值
    metal_boundary = np.zeros_like(radon_metal)
    for projection_idx in range(radon_metal.shape[1]):
        for distance_idx in range(1, radon_metal.shape[0] - 1):
            if radon_metal[distance_idx, projection_idx] > 0:
                if (radon_metal[distance_idx + 1, projection_idx] > 0 and
                    radon_metal[distance_idx - 1, projection_idx] == 0) or \
                        (radon_metal[distance_idx + 1, projection_idx] == 0 and
                         radon_metal[distance_idx - 1, projection_idx] > 0):
                    metal_boundary[distance_idx, projection_idx] = 100

    for projection_idx in range(radon_metal.shape[1]):
        upper_index = -1
        for distance_idx in range(radon_metal.shape[0]):
            if metal_boundary[distance_idx, projection_idx] == 100:
                if upper_index == -1:
                    upper_index = distance_idx
                else:
                    midpoint = (upper_index + distance_idx) // 2
                    if radon_metal[midpoint, projection_idx] > 0:
                        for k in range(upper_index, distance_idx + 1):
                            weight = (k - upper_index) / (distance_idx - upper_index)
                            radon_non_metal[k, projection_idx] = (1 - weight) * radon_non_metal[
                                upper_index, projection_idx] + \
                                                                 weight * radon_non_metal[distance_idx, projection_idx]
                        upper_index = -1
                    else:
                        upper_index = distance_idx

    # 4. 逆Radon变换
    reconstructed_image = iradon(radon_non_metal, theta=projection_angles, circle=False, filter_name=FILTER)

    # 调整尺寸到原始图像大小
    if reconstructed_image.shape != image_array.shape:
        reconstructed_image = cv2.resize(reconstructed_image, (image_array.shape[1], image_array.shape[0]))

    # 5. 金属补偿和归一化
    max_intensity = np.max(reconstructed_image)
    for i in range(reconstructed_image.shape[0]):
        for j in range(reconstructed_image.shape[1]):
            if metal_mask[i, j] > 0:
                reconstructed_image[i, j] = max_intensity

    # 6. 归一化到0-1范围并缩放到0-255
    normalized_image = (reconstructed_image - np.min(reconstructed_image)) / \
                       (np.max(reconstructed_image) - np.min(reconstructed_image))
    result_image = (normalized_image * 255).astype(np.uint8)

    return result_image

if __name__=='__main__':
    original_image = cv2.imread('./data/9FdkRecon-ushort-300x300x300.view.raw.to-byte299.png', cv2.IMREAD_GRAYSCALE)
    # 调用函数处理图像
    output_image = remove_metal_artifacts(original_image)

    # 保存结果
    cv2.imwrite('output.png', output_image)

    # 显示结果对比
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(original_image, cmap='gray')
    plt.title('origin')
    plt.subplot(122)
    plt.imshow(output_image, cmap='gray')
    plt.title('result')
    plt.show()