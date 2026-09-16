import os
import numpy as np
from PIL import Image

def normalize_to_uint8(data):
    """将数据归一化到0-255并转换为uint8类型"""
    if data.dtype.kind in ('f', 'd'):
        # 处理浮点型数据
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min == 0:
            return np.zeros_like(data, dtype=np.uint8)
        data = (data - data_min) / (data_max - data_min) * 255
    else:
        # 处理整数类型数据，假设范围在0-255之间，否则可能溢出
        data = data.astype(np.uint16)  # 防止溢出，例如uint16转uint8
        data = (data / (np.max(data) / 255)).astype(np.uint8)
    return data.astype(np.uint8)

def convert_npy_to_png(npy_path):
    try:
        data = np.load(npy_path)
        # 处理数据
        data = normalize_to_uint8(data)

        # 根据维度创建图像
        if len(data.shape) == 2:
            img = Image.fromarray(data, 'L')
        elif len(data.shape) == 3:
            # 检查通道数
            if data.shape[2] == 3:
                img = Image.fromarray(data, 'RGB')
            elif data.shape[2] == 4:
                img = Image.fromarray(data, 'RGBA')
            else:
                # 其他通道数转为灰度（取第一个通道）
                img = Image.fromarray(data[:, :, 0], 'L')
        else:
            print(f"跳过 {npy_path}：不支持的维度 {data.shape}")
            return

        # 保存为PNG
        png_path_x = os.path.splitext(npy_path)[0].split('/')[-1]
        png_path = "" + png_path_x + '.png'
        img.save(png_path)
        print(f"转换成功：{png_path}")
    except Exception as e:
        print(f"处理 {npy_path} 时出错：{str(e)}")

def process_directory(root_dir):
    for foldername, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npy'):
                npy_path = os.path.join(foldername, filename)
                convert_npy_to_png(npy_path)

if __name__ == "__main__":
    process_directory("")