import numpy as np
from cv2 import cv2
import glob

# 设置终止条件，迭代30次或变动小于0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 生成42×3的矩阵，用来保存棋盘图中6*7个内角点的3D坐标，也就是物体点坐标
objp = np.zeros((6*7, 3), np.float32)

# 通过np.mgrid生成对象的xy坐标点
# 最终得到的objp为(0,0,0), (1,0,0), (2,0,0) ,..., (6,5,0)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

obj_points = []  # 用于保存物体点
img_points = []  # 用于保存图像点

# 返回当前目录所有匹配的jpg图片
images = glob.glob('*.jpg')

for fname in images:
    # 读取图片
    img = cv2.imread(fname)
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 寻找棋盘图的内角点位置
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    print(corners)

    # 如果找到棋盘图的所有内角点
    if ret == True:

        obj_points.append(objp)
        # 亚像素级角点检测，在角点检测中精确化角点位置
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)

        # 在图中标注角点，方便查看结果
        img = cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        cv2.imshow('img', img)
        # cv2.imwrite('img.jpg', img)
        cv2.waitKey(500)
        # break

cv2.destroyAllWindows()

# 相机标定，返回相机矩阵、畸变系数、旋转向量和平移向量
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape, None, None)

# 读入图片
img = cv2.imread('left12.jpg')
# 获取图片的长宽
h,  w = img.shape[:2]

# 根据尺度因子调节相机矩阵
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

'''
# 校正畸变图片方法一
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 裁剪图片
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)
'''
# 校正畸变图片方法二
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# 裁剪图片
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)

# 计算重投影误差
mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2)/len(img_points2)
    mean_error += error

print("total error: ", mean_error/len(obj_points))