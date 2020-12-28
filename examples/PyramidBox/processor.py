import os
import cv2
import numpy as np

__all__ = ['preprocess', 'postprocess']

def preprocess(img_path, shrink):
    image = cv2.imread(img_path)
    image_height, image_width, image_channel = image.shape
    if shrink != 1:
        image_height, image_width = int(image_height * shrink), int(
            image_width * shrink)
        image = cv2.resize(image, (image_width, image_height),
                           cv2.INTER_NEAREST)
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # mean, std
    mean = [104., 117., 123.]
    scale = 0.007843
    image = image.astype('float32')
    image -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
    image = image * scale
    image = np.expand_dims(image, axis=0).astype('float32')
    return image

def postprocess(output_datas, output_dir, img_path, model_name):
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    for label, score, x1, y1, x2, y2 in output_datas:
        if score>0.9:
            x1, y1, x2, y2 = [int(_) for _ in [x1*img_w, y1*img_h, x2*img_w, y2*img_h]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)   

    # 检查输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cv2.imwrite(os.path.join(output_dir, '%s.jpg' % model_name), img)