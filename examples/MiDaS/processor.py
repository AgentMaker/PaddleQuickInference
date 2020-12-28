import os
import cv2
import numpy as np
from utils import write_depth
from paddle.vision.transforms import Compose
from transforms import Resize, NormalizeImage, PrepareForNet

# 数据预处理函数
def preprocess(img_path, size):
    # 图像变换
    transform = Compose([
        Resize(
            size,
            size,
            resize_target=None,
            keep_aspect_ratio=False,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

    # 读取图片
    img = cv2.imread(img_path)

    # 归一化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    # 图像变换
    img = transform({"image": img})["image"]

    # 新增维度
    img = img[np.newaxis, ...]

    return img

# 数据后处理函数
def postprocess(results, output_dir, img_path, model_name):
    # 检查输出目录
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 读取输入图像
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 缩放回原尺寸
    output = cv2.resize(results[0], (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 可视化输出
    pfm_f, png_f = write_depth(os.path.join(output_dir, model_name), output, bits=2)