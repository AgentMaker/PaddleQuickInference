import os
import cv2
import numpy as np

def preprocess(img_path):
    # 读取图片
    img = cv2.imread(img_path)
    
    # 缩放图像
    img = cv2.resize(img, (256, 256))

    # 归一化
    img = (img / 255.0 - 0.5) / 0.5

    # 转置
    img = img.transpose((2, 0, 1))

    # 增加维度
    img = np.expand_dims(img, axis=0).astype('float32')
    
    # 返回输入数据
    return img

def postprocess(outputs, output_dir, model_name):
    # 反归一化
    img = (outputs[0] * 0.5 + 0.5) * 255.
    
    # 限幅
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # 转置
    img = img.transpose((1, 2, 0))

    # 检查输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 写入输出图片
    cv2.imwrite(os.path.join(output_dir, '%s.jpg' % model_name), img)