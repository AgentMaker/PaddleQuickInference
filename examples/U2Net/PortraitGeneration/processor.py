import os
import cv2
import numpy as np

__all__ = ['preprocess', 'postprocess']

def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def preprocess(img_path):
    img = cv2.imread(img_path)

    # 缩放
    img = cv2.resize(img, (512, 512))

    # 归一化
    means = [0.225, 0.224, 0.229]
    vars = [0.406, 0.456, 0.485]
    img = img/np.max(img)
    img -= vars
    img /= means

    # convert BGR to RGB
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :].astype('float32')

    return img


def postprocess(outputs, output_dir, model_name):
    pred = 1.0 - outputs[:, 0, :, :]
    pred = normPRED(pred)
    pred = pred.squeeze()
    pred = (pred*255).astype(np.uint8)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    cv2.imwrite(os.path.join(output_dir, '%s.jpg' % model_name), pred)
