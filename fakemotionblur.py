# coding:utf-8
import numpy as np
from cv2 import cv2
import math
import os
import time
from tqdm import tqdm
import random

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

def blurKernel(length, angle):
    # 自制模糊核，长度和角度
    # 先假设y的正方向为上，需要时再翻转
    half = length / 2
    alpha = angle / 180.0 * math.pi
    cosalpha = math.cos(alpha)  
    sinalpha = math.sin(alpha)  
    width_2 = int(half * math.fabs(cosalpha))
    height_2 = int(half * math.fabs(sinalpha))
    size_w = 2 * width_2 + 1
    size_h = 2 * height_2 + 1
    center_x = width_2 + 1
    center_y = height_2 + 1
    total = 0
    biggerthan45 = math.fabs(math.tan(alpha)) > 1
    kernel = np.zeros((size_h, size_w))
    for x in range(0, size_w):
        for y in range(0, size_h):
            realx = x - center_x
            realy = y - center_y
            powe = 1
            if biggerthan45:
                calx = realy / sinalpha * cosalpha
                value = powe if math.fabs(calx - realx) <= 2 else 0
            else:
                caly = realx / cosalpha * sinalpha
                value = powe if math.fabs(caly - realy) <= 2 else 0
            if value > 0:
                total += value
            kernel[y,x] = value
    if total > 0:
        kernel = kernel / total
    kernel = np.flipud(kernel) # 翻转kernel为正
    return kernel

def testBlur():
    img = cv2.imread('resource/one1.png')
    for ang in range(-360, 360, 15):
        kernel=blurKernel(length = 20, angle = ang)
        motion_blur=cv2.filter2D(img,-1,kernel)
        # blur = cv2.GaussianBlur()
        winname = 'test motion {}'.format(ang)
        cv2.imshow(winname, motion_blur)
        cv2.waitKey(0)
        cv2.destroyWindow(winname)

def drawRegionBlur(src, dsc, region, mag, ang, maskvalue = 1): # this will change dsc
    starty, endy, startx, endx = region
    rows, cols, _ = src.shape
    if mag < 2:
        dsc[starty:endy, startx:endx] = src[starty:endy, startx:endx]
        return dsc
    kernel = blurKernel(length = mag, angle = ang)
    krows, kcols = kernel.shape
    blur_startx = startx - kcols if startx - kcols > 0 else 0
    blur_endx = endx + kcols if endx + kcols < cols else cols
    blur_starty = starty - krows if starty - krows > 0 else 0
    blur_endy = endy + krows if endy + krows < rows else rows

    blur_roi = cv2.filter2D(src[blur_starty:blur_endy, blur_startx:blur_endx],-1,kernel)
    
    blurmask = np.zeros((rows, cols, 1))
    blurmask[starty:endy, startx:endx] = maskvalue
    blur_roi_mask = cv2.filter2D(blurmask[blur_starty:blur_endy, blur_startx:blur_endx], -1, kernel)
    
    # draw the blur
    dsc_roi = dsc[blur_starty:blur_endy, blur_startx:blur_endx]
    locations = np.where(blur_roi_mask > 0)
    val = blur_roi_mask[locations]
    val[val > 1] = 1
    val = val.reshape(-1, 1)
    dsc_roi[locations] = dsc_roi[locations] * (1 - val) + blur_roi[locations] * val
    dsc[blur_starty:blur_endy, blur_startx:blur_endx] = dsc_roi
    return dsc


test_round_x = 36 #40
test_round_y = int(test_round_x / 2)
shouldprintblurprogress = False
# round_x_progress = tqdm(range(test_round_x), leave=True)

def roughlyBlur(src, dsc, allflow, fps = 30):
    # print("roughly blur drawing")
    scale = 1.0
    if fps > 30:
        scale = float(fps) / 30.0
    rows, cols, _ = src.shape
    test_width = int(cols / test_round_x)
    test_height = int(rows / test_round_y)
    test_x = 0
    # round_x_progress.reset(total=test_round_x)

    # load all roi's mag and ang
    arr = []
    while test_x < cols: #round_x_progress:
        # round_x_progress.set_description('roughly blur drawing')
        startx = test_x
        endx = startx + test_width if startx + test_width < cols else cols
        test_x = endx
        if startx == endx:
            break 
        test_y = 0

        while test_y < rows:
            starty = test_y
            endy = starty + test_height if starty + test_height < rows else rows
            test_y = endy
            if starty == endy:
                break

            flow_roi = allflow[starty:endy, startx:endx]
            flow_avg_0 = np.array((np.average(flow_roi[...,0])))
            flow_avg_1 = np.array((np.average(flow_roi[...,1])))
            mag, ang = cv2.cartToPolar(flow_avg_0, flow_avg_1)
            mag = (mag * scale) #/ 2.0
            ang = -ang * 180.0 / math.pi # 这原本是弧度！
            mag_using = mag[0,0]
            ang_using = ang[0,0]
            region = (starty, endy, startx, endx)

            item = (region, mag_using, ang_using)
            arr.append(item)
    sortedarr = sorted(arr, key=lambda iii: iii[1], reverse=True)
    count = len(sortedarr)
    for index in range(count):
        if shouldprintblurprogress and (random.randint(1,50) % 10) == 0:
            print("test", index, count)
        region, mag, ang = sortedarr[index]
        if math.isinf(mag):
            continue
        drawRegionBlur(src=src, dsc=dsc, region=region, mag=mag, ang=ang, maskvalue=1.5)
    return dsc

def flowFarnebackWithFrames(frame1, frame2, fps = 30):
    prevF = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    nextF = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    _, cols, _ = frame1.shape
    somesize = int(cols / 30)
    # flow 为每一个像素点的偏移
    flow = cv2.calcOpticalFlowFarneback(prev=prevF, next=nextF, flow=None, pyr_scale=0.5, levels=10, winsize=somesize, iterations=10, poly_n=5, poly_sigma=1.2, flags=0)
    res = frame1.copy()
    res = roughlyBlur(src=frame1, dsc=res, allflow=flow, fps=fps)
    return res

def testImages():
    frame1 = cv2.imread('resource/fly1.jpg')
    frame2 = cv2.imread('resource/fly2.jpg')
    # res = motionBlurWithFrames(frame1, frame2)
    # cv2.imshow('blr', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # test
    imgsize = (1920, 1080)
    frame1 = cv2.resize(frame1, imgsize)
    frame2 = cv2.resize(frame2, imgsize)
    # res = flowLucasWithFrames(frame1, frame2)
    res = flowFarnebackWithFrames(frame1, frame2, fps = 30)
    # res = simpleAddWithFrames(frame1, frame2)

    name = 'blur'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(name, 640, 480)
    cv2.imshow(name, res)
    cv2.waitKey(0)
    cv2.destroyWindow(name)

if __name__ == "__main__":
    shouldprintblurprogress = True
    # blurVideo()
    testImages()
    # testBlur()