import os
import cv2

__all__ = ['preprocess', 'postprocess']

def preprocess(img_path):
    inHeight = 368
    img = cv2.imread(img_path)
    img_height, img_width, _ = img.shape
    aspect_ratio = img_width / img_height
    inWidth = int(((aspect_ratio * inHeight) * 8) // 8)
    inpBlob = cv2.dnn.blobFromImage(
        img, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    
    return inpBlob

# 结果后处理函数
def postprocess(outputs, output_dir, img_path, model_name, threshold):
    img = cv2.imread(img_path)
    num_points = 21
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 结果后处理
    points = []
    for idx in range(num_points):
        probMap = outputs[0, idx, :, :]
        img_height, img_width, _ = img.shape
        probMap = cv2.resize(probMap, (img_width, img_height))
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold:
            points.append([int(point[0]), int(point[1])])
        else:
            points.append(None)

    # 结果可视化
    vis_pose(img, output_dir, points, model_name)

# 结果可视化
def vis_pose(img, output_dir, points, model_name):
    point_pairs = [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], 
        [6, 7], [7, 8], [0, 9], [9, 10], [10, 11],
        [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], 
        [0, 17], [17, 18], [18, 19], [19, 20]
    ]

    # 根据结果绘制关键点到原图像上
    for pair in point_pairs:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(img, tuple(points[partA]), tuple(points[partB]), (0, 255, 255), 3)
            cv2.circle(img, tuple(points[partA]), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    # 可视化图像保存
    cv2.imwrite(os.path.join(output_dir, '%s.jpg' % model_name), img)