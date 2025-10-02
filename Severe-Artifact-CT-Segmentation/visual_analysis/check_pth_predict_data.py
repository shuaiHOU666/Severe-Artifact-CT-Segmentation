import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
from skimage.transform import radon, iradon
from tifffile import tifffile
import cv2
import matplotlib.pyplot as plt


# intro net ;
from MODULES.adapter import dinov2_mla

device = torch.device('cuda:0')

n1, n2 = 1250, 1250
classes = 3
patch_h = 89
patch_w = 89

import torchvision.transforms as T
from PIL import Image
# Create the transform



target_size = (patch_h * 14, patch_w * 14)


def process_tensor(data):
    # Convert tensor to numpy array and remove batch/channel dims for PIL
    print(data.shape)
    img = Image.fromarray(np.uint16(data))# Now shape [1250, 1250]

    # Initialize output tensor (batch_size=2, channel=1, H, W)
    img_tensor = torch.zeros([1,2, *target_size], dtype=torch.float32)

    # Apply transform to both batch elements

    transform = T.Compose([
        T.Resize((patch_h * 14, patch_w * 14)),
        T.ToTensor(),
    ])

    transformed = transform(img)
    img_tensor[0, 0] = transformed
    img_tensor[0, 1] = transformed  # Duplicate for batch dimension

    # import pdb;pdb.set_trace()
    # Convert to 3 channels by repeating along channel dim

    img_tensor = img_tensor.unsqueeze(2).repeat(1, 1, 3, 1, 1)

    return img_tensor

model_Path = '255-710-20-core_seg/wdice/mla/lora_small_maxiou_train.pth'
lora_Path = '255-710-20-core_seg/wdice/mla/lora_small_maxiou_train_lora.pth'
pngPath = '../png/'

predictPath = os.path.join(pngPath, 'prediction')
if not os.path.exists(predictPath):
    os.makedirs(predictPath)

cmin = 0
cmax = classes - 1

net = dinov2_mla(classes, pretrain=True)
net.load_state_dict(torch.load(model_Path, map_location=device), strict=False)
net.load_state_dict(torch.load(lora_Path, map_location=device), strict=False)

net.to(device)
net.eval()


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


def process_raw_slices(input_path,input2_path, output_dir):
    """
    处理RAW文件中的每一张切片
    :param input_path: 输入RAW文件路径
    :param output_dir: 输出目录
    :param start_slice: 开始处理的切片索引
    :param end_slice: 结束处理的切片索引
    """
    # 读取原始数据
    data_3d = np.fromfile(input_path, dtype=np.uint16).reshape((1100, 1250, 1250))
    process_data_3d = tifffile.imread(input2_path)
    # 根据文件路径判断形状
    # if '1250x1250x1100' in input_path:
    #     data_3d = data_3d
    # elif '1850x1850x1100' in input_path:
    #     data_3d = data_3d.reshape((1100, 1850, 1850))
    # else:
    #     raise ValueError("无法确定RAW文件的形状，请检查文件名是否符合约定")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(os.path.join(output_dir, "predict"), exist_ok=True)


    # process_result = np.zeros_like(data_3d)

    predict_process_result = np.zeros_like(data_3d)

    print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"保留内存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
    # 处理每一张切片
    for i in range(data_3d.shape[0]):

        print(f"Processing slice {i + 1}/{data_3d.shape[0]}")
        torch.cuda.empty_cache()
        # 获取当前切片并归一化到0-255

        current_slice = data_3d[i]

        process_slice = process_data_3d[i]

        normalized_slice = ((current_slice - np.min(current_slice)) /
                            (np.max(current_slice) - np.min(current_slice)) * 255)
        normalized_slice = normalized_slice.astype(np.uint8)


        # 保存原始切片

        # origin_path = os.path.join(output_dir, "origin", f"slice_{i:04d}.tiff")
        # tifffile.imwrite(origin_path, normalized_slice)
        # cv2.imwrite(origin_path, normalized_slice)




        data = process_tensor(normalized_slice)


        b1, b2, c, h, w = data.shape

        data = data.to(device).reshape(b1 * b2, c, h, w)
        import pdb;pdb.set_trace()
        # 使用混合精度训练
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            _, preds = torch.max(net(data, (n1, n2)), 1)
        predict_arr = np.squeeze(preds[0].detach().cpu().numpy()).astype(np.uint8)
        torch.cuda.empty_cache()
        plt.figure(figsize=(12, 6))

        plt.subplot(131)
        plt.imshow(normalized_slice, cmap='gray')
        plt.title(f'Origin Slice {i}')
        plt.subplot(132)
        plt.imshow(process_slice, cmap='jet')
        plt.title(f'Processed Slice {i}')
        plt.subplot(133)
        plt.imshow(predict_arr, cmap='jet')
        plt.title(f'Processed Slice {i}')
        plt.show()
        import pdb;pdb.set_trace()
        res_path = os.path.join(output_dir, "predict", f"predict_slice_{i:04d}.tiff")
        normalized_image = (predict_arr - np.min(predict_arr)) / \
                           (np.max(predict_arr) - np.min(predict_arr))
        # result_image = (normalized_image * 255).astype(np.uint8)

        # tifffile.imwrite(res_path, result_image)
        #
        # predict_process_result[i] = result_image

        # import pdb;pdb.set_trace()


        # cv2.imwrite(processed_path, processed_slice)

        # 每处理10张切片显示一次对比
        if i % 1 == 0:
            plt.figure(figsize=(12, 6))

            plt.subplot(131)
            plt.imshow(normalized_slice, cmap='gray')
            plt.title(f'Origin Slice {i}')
            plt.subplot(132)
            plt.imshow(process_slice, cmap='jet')
            plt.title(f'Processed Slice {i}')
            plt.subplot(133)
            plt.imshow(predict_arr, cmap='jet')
            plt.title(f'Processed Slice {i}')
            plt.show()
            # plt.savefig(os.path.join(output_dir, "predict", f"comparision_{i:04d}.png"))
            # plt.close()

    # import pdb;pdb.set_trace()
    # tifffile.imwrite('process_result.tiff', process_result.astype(np.uint8))

    tifffile.imwrite('predict_process_result_eold.tiff', predict_process_result.astype(np.uint8)* 127)

    # process_result.save #  !!!! save as raw or tiff

if __name__ == '__main__':
    # 处理RAW文件中的所有切片

    input_raw_file = './data&label/o_data.raw'
    input_process_data ='./data&label/p_data.tiff'
    output_directory = 'predict_old'


    # input is data shape is (1,1250,1250) array

    # 可以设置只处理部分切片，例如从第100到200切片:
    # process_raw_slices(input_raw_file, output_directory, start_slice=100, end_slice=200)
    # 处理所有切片

    process_raw_slices(input_raw_file,input_process_data, output_directory)