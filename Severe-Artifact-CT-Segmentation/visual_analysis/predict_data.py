import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from tifffile import tifffile

def normalize_data(data, dtype=np.uint16):
    """归一化数据至 [0, 1] 或 [0, 65535] 范围，并转换为指定类型"""
    norm_data = data.astype(np.float32)
    norm_data -= norm_data.min()  # 归一化至 [0, max]
    norm_data /= norm_data.max()  # 归一化至 [0, 1]
    return norm_data.astype(dtype)  # 转换为指定类型（uint16）

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP冲突
if __name__ == '__main__':
    # 创建输出目录
    output_dir = '7-17-res_visual'
    os.makedirs(output_dir, exist_ok=True)

    # 创建数据保存目录
    save_data_dir = './valid_data'
    os.makedirs(save_data_dir, exist_ok=True)

    file3 = 'converted_predict_process_result_e50.tiff'
    file4 = 'converted_7-16-predict_direct_label.tiff'
    file5_1 = 'process_rock_volume_1250.labels.raw'
    file5 = 'process_rock_volume_uint8.tiff'
    file6 = 'origin_data_9FdkRecon-ushort-1250x1250x1100.view.raw'
    file7 = 'origin_witharfactlabel_9FdkRecon-ushort-1250x1250x2.labels.raw'

    # 读取数据
    ori_data = np.fromfile(file6, dtype=np.uint16).reshape((1100, 1250, 1250))
    ori_label = np.fromfile(file7, dtype=np.uint8).reshape((1100, 1250, 1250))
    ori_label = np.select(
        [ori_label == 1, ori_label == 2],
        [2, 1],
        ori_label
    )
    process_data = tifffile.imread(file5)
    process_label = np.fromfile(file5_1, dtype=np.uint8).reshape((1100, 1250, 1250))

    # 读取TIFF文件
    res1 = tifffile.imread(file4)  # from processed
    res2 = tifffile.imread(file3)  # from origin

    # 归一化数据
    norm_ori_data = normalize_data(ori_data, dtype=np.uint16)
    norm_process_data = normalize_data(process_data, dtype=np.uint16)

    # 保存数据为TIFF和RAW格式

    # 保存归一化后的数据
    tifffile.imwrite(os.path.join(save_data_dir, 'core_ori.tiff'), norm_ori_data)
    norm_ori_data.tofile(os.path.join(save_data_dir, 'core_ori.raw'))

    tifffile.imwrite(os.path.join(save_data_dir, 'core_process.tiff'), norm_process_data)
    norm_process_data.tofile(os.path.join(save_data_dir, 'core_process.raw'))

    # 修改 label 的 raw 类型为 uint8 并保存
    tifffile.imwrite(os.path.join(save_data_dir, 'label_ori.tiff'), ori_label.astype(np.uint8))
    ori_label.astype(np.uint8).tofile(os.path.join(save_data_dir, 'label_ori.raw'))

    tifffile.imwrite(os.path.join(save_data_dir, 'label_o2o.tiff'), process_label.astype(np.uint8))
    process_label.astype(np.uint8).tofile(os.path.join(save_data_dir, 'label_o2o.raw'))

    tifffile.imwrite(os.path.join(save_data_dir, 'label_p2p.tiff'), res2.astype(np.uint8))
    res2.astype(np.uint8).tofile(os.path.join(save_data_dir, 'label_p2p.raw'))

    tifffile.imwrite(os.path.join(save_data_dir, 'label_o2p.tiff'), res1.astype(np.uint8))
    res1.astype(np.uint8).tofile(os.path.join(save_data_dir, 'label_o2p.raw'))
#
#     # 创建说明文档
#     with open(os.path.join(save_data_dir, 'description.txt'), 'w') as f:
#         f.write("""
# ori_label (label_ori): 直接分的结果（伪影干扰孔隙结果）
# process_label (label_o2o): 直接原始分割（存在输入origin,输出origin）；存在伪影，最好的效果接近手动标签，有伪影
# res2 (label_p2p): 驱除伪影之后直接进行分割 (输入process, 输出process); 伪影驱除了，但是孔隙空间破坏掉了，只留下矿物；还带来了一些边缘上错误的矿物
# res1 (label_o2p): 最终的最好的结果，采用了我们的策略（输入origin，输出process）；不仅驱除了伪影，同时又保留了孔隙结构
#         """)

    # 生成可视化比较图像
    for i in range(0, 1100, 11):  # 每11个切片取一个
        print(f'{i}/{1100}')
        plt.figure(figsize=(20, 10))

        # Original data and label
        plt.subplot(2, 3, 1)
        plt.imshow(ori_data[i], cmap='gray')
        plt.title(f'Original Data\nSlice {i}')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(ori_label[i], cmap='jet')
        plt.title('Original Label')
        plt.axis('off')

        # Processed data and label
        plt.subplot(2, 3, 3)
        plt.imshow(process_data[i], cmap='gray')
        plt.title('Processed Data')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(process_label[i], cmap='jet')
        plt.title('Processed Label')
        plt.axis('off')

        # Results
        plt.subplot(2, 3, 5)
        plt.imshow(res2[i], cmap='jet')
        plt.title('Result (from processed)')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(res1[i], cmap='jet')
        plt.title('Result (from origin)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        # 保存图像
        # save_path = os.path.join(output_dir, f'comparison_slice_{i:04d}.png')
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f'Saved visualization for slice {i} to {save_path}')

    print("All processing completed!")