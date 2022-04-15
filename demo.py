#-*- coding: utf-8 -*-

# 通过OpenCV实现车道线检测

# Key Point:
# 1.打开视频文件
# 2.循环遍历每一帧
# 3.canny边缘检测，检测line
# 4.去除多余图像直线
# 5.霍夫变换
# 6.叠加变换与原始图像
# 7.车道检测

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Global Variables
IMG_LOCATION = "d.jpg"

# Tools
# Canny检测


def do_canny(frame):
    # 将每一帧转化为灰度图像，去除多余信息
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 高斯滤波器，去除噪声，平滑图像
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # 边缘检测
    # minVal = 50
    # maxVal = 150
    canny = cv.Canny(blur, 50, 100)

    return canny

# 图像分割，去除多余线条信息


def do_segment(frame):
    # 获取图像高度(注意CV的坐标系,正方形左上为0点，→和↓分别为x,y正方向)
    height = frame.shape[0]
    width = frame.shape[1]

    # 创建一个三角形的区域,指定三点
    polygons = np.array([
        [(20, height - 20),
         (width - 20, height - 20),
         (width - 20, 20),
         (width // 2, height - 20),
         (20, 20)]
    ])

    # 创建一个mask,形状与frame相同，全为0值
    mask = np.zeros_like(frame)

    # 对该mask进行填充，做一个掩码
    # 三角形区域为1
    # 其余为0
    cv.fillPoly(mask, polygons, 255)

    # 将frame与mask做与，抠取需要区域
    segment = cv.bitwise_and(frame, mask)

    return segment

# 车道左右边界标定


def calculate_lines(frame, lines):
    # 建立两个空列表，用于存储左右车道边界坐标
    left = []
    right = []

    if lines is None:
        return None
    # 循环遍历lines
    for line in lines:
        # 将线段信息从二维转化能到一维
        x1, y1, x2, y2 = line.reshape(4)

        # 将一个线性多项式拟合到x和y坐标上，并返回一个描述斜率和y轴截距的系数向量
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]  # 斜率
        y_intercept = parameters[1]  # 截距

        # 通过斜率大小，可以判断是左边界还是右边界
        # 很明显左边界slope<0(注意cv坐标系不同的)
        # 右边界slope>0
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
        # if slope > 0:
        #     right.append((slope, y_intercept))

    # 将所有左边界和右边界做平均，得到一条直线的斜率和截距
    left_avg = np.average(left, axis=0) if left else [0, 0]
    right_avg = np.average(right, axis=0) if right else [0, 0]
    # print(left_avg, right_avg)
    if abs(left_avg[0]) < 0.2 and abs(right_avg[0]) < 0.2:
        return None
    elif abs(left_avg[0]) < 0.2:
        left_avg = [0, 0]
    elif abs(right_avg[0]) < 0.2:
        right_avg = [0, 0]

    # 将这个截距和斜率值转换为x1,y1,x2,y2
    if abs(left_avg[0]) > abs(right_avg[0]):
        line_axis = calculate_coordinate(frame, parameters=left_avg)
        line_axis_v = calculate_coordinate(
            frame, parameters=[-0.2 / left_avg[0], frame.shape[0] * 0.5])
    else:
        line_axis = calculate_coordinate(frame, parameters=right_avg)
        line_axis_v = calculate_coordinate(
            frame, parameters=[-0.75 / right_avg[0], frame.shape[0] * 0.95])

    # right_avg = np.average(right, axis=0) if right else [0, 0]
    # print(right_avg[0])
    # if abs(right_avg[0]) < 0.2:
    #     return None
    # line_axis = calculate_coordinate(frame, parameters=right_avg)
    # line_axis_v = calculate_coordinate(
    #     frame, parameters=[-0.75 / right_avg[0], frame.shape[0] * 0.95])

    return np.array([line_axis, line_axis_v])

# 将截距与斜率转换为cv空间坐标


def calculate_coordinate(frame, parameters):
    # 获取斜率与截距
    slope, y_intercept = parameters

    # 设置初始y坐标为自顶向下(框架底部)的高度
    # 将最终的y坐标设置为框架底部上方150
    y1 = frame.shape[0]
    y2 = 0   # int(y1 - 300)
    # 根据y1=kx1+b,y2=kx2+b求取x1,x2
    x1 = int((y1 - y_intercept) / slope)
    x2 = int((y2 - y_intercept) / slope)
    return np.array([x1, y1, x2, y2])

# 可视化车道线


def visualize_lines(frame, line_axis):
    lines_visualize = np.zeros_like(frame)
    # 检测lines是否为空
    if line_axis is not None:
        # 画线
        # cv.line(lines_visualize, (line_axis[0], line_axis[1]),
        #         (line_axis[2], line_axis[3]), (0, 0, 255), 5)
        for line in line_axis:
            cv.line(lines_visualize, (line[0], line[1]),
                    (line[2], line[3]), (0, 0, 255), 5)
    return lines_visualize


if __name__ == "__main__":

    # 图片读取
    frame = cv.imread(IMG_LOCATION)

    # 边缘检测
    canny = do_canny(frame)
    # cv.namedWindow("canny", cv.WINDOW_NORMAL)
    # cv.imshow("canny", canny)

    # 图像分割，去除多余直线,只保留需要的直线
    # 原理见博文
    segment = do_segment(canny)
    cv.namedWindow("segment", cv.WINDOW_NORMAL)
    cv.imshow("segment", segment)

    # 原始空间中，利用Canny梯度，找到很多练成线的点
    # 利用霍夫变换，将这些点变换到霍夫空间中，转换为直线
    hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100,
                           minLineLength=110, maxLineGap=4)
    # cv.imshow("hough", hough)

    # 将从hough检测到的多条线平均成一条线表示车道的左边界，
    # 一条线表示车道的右边界
    line_axis = calculate_lines(frame, hough)

    # 可视化
    lines_visualize = visualize_lines(frame, line_axis)  # 显示
    # cv.imshow("lines",lines_visualize)

    # 叠加检测的车道线与原始图像,配置两张图片的权重值
    # alpha=0.6, beta=1, gamma=1
    output = cv.addWeighted(frame, 0.6, lines_visualize, 1, 0.1)
    cv.namedWindow("output", cv.WINDOW_NORMAL)
    cv.imshow("output", output)
    cv.waitKey(0)
