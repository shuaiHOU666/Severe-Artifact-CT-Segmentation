import os
import numpy as np
import tifffile
from natsort import natsorted

input_directory = "./processed"
output_file = "process_rock_volume_uint16.tiff"

all_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.tiff')]
sorted_files = natsorted(all_files)

res_data = np.zeros((1100, 1250, 1250), dtype=np.uint16)

for i, f in enumerate(sorted_files):
    filepath = os.path.join(input_directory, f)
    try:
        print(f'Processing {i+1}/{len(sorted_files)}: {f}')
        data = tifffile.imread(filepath)
        if data.shape != (1250, 1250):
            print(f"警告：{f} 尺寸不符，应为(1250,1250)，实际为{data.shape}")
            continue
        res_data[i] = data
    except Exception as e:
        print(f"无法读取 {f} : {str(e)}")
        # 用空白切片替代或终止处理
        res_data[i] = np.zeros((1250, 1250), dtype=np.uint16)

tifffile.imwrite(output_file, res_data)
print(f"完成！已保存到 {output_file}")