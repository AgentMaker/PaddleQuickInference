import os
import cv2
import numpy as np

__all__ = ['preprocess', 'postprocess']

mean = [107.304565, 115.69884, 132.35703 ]
std = [63.97182, 65.1337, 68.29726]

# 预处理函数
def preprocess(img_path):
    # 读取图像
    img = cv2.imread(img_path)

    # 缩放
    h, w = img.shape[:2]
    img = cv2.resize(img, (224, 224))

    # 格式转换
    img = img.astype(np.float32)

    # 归一化
    for j in range(3):
        img[:, :, j] -= mean[j]
    for j in range(3):
        img[:, :, j] /= std[j]
    img /= 255.

    # 格式转换
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, ...]

    return img

# 后处理函数
def postprocess(results, output_dir, img_path, model_name):
    # 检查输出目录
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # 读取图像
    img = cv2.imread(img_path)

    # 计算MASK
    mask = (results[0][0] > 0).astype('float32')

    # 缩放
    h, w = img.shape[:2]
    mask = cv2.resize(mask, (w, h))

    # 计算输出图像
    result = (img * mask[..., np.newaxis] + (1 - mask[..., np.newaxis]) * 255).astype(np.uint8)

    # 格式还原
    mask = (mask * 255).astype(np.uint8)

    # 可视化
    cv2.imwrite(os.path.join(output_dir, 'result_mask_%s.png' % model_name), mask)
    cv2.imwrite(os.path.join(output_dir, 'result_%s.png' % model_name), result)