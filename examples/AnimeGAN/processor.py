import os
import cv2
import numpy as np

__all__ = ['preprocess', 'postprocess']

def preprocess(img_path, max_size=512, min_size=32):
    # 读取图片
    img = cv2.imread(img_path)

    # 格式转换
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 缩放图片
    h, w = img.shape[:2]
    if max(h,w)>max_size:
        img = cv2.resize(img, (max_size, int(h/w*max_size))) if h<w else cv2.resize(img, (int(w/h*max_size), max_size))
    elif min(h,w)<min_size:
        img = cv2.resize(img, (min_size, int(h/w*min_size))) if h>w else cv2.resize(img, (int(w/h*min_size), min_size))

    # 裁剪图片
    h, w = img.shape[:2]
    img = img[:h-(h%32), :w-(w%32), :]

    # 归一化
    img = img/127.5 - 1.0

    # 新建维度
    img = np.expand_dims(img, axis=0).astype('float32')
    
    # 返回输入数据
    return img

def postprocess(output, output_dir, model_name):
    # 反归一化
    image = (output.squeeze() + 1.) / 2 * 255

    # 限幅
    image = np.clip(image, 0, 255).astype(np.uint8)

    # 格式转换
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 检查输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 写入输出图片
    cv2.imwrite(os.path.join(output_dir, '%s.jpg' % model_name), image)