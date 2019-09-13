# coding:utf-8
import numpy as np
from cv2 import cv2
import math
import os

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
            if biggerthan45:
                calx = realy / sinalpha * cosalpha
                value = 1 if math.fabs(calx - realx) <= 2 else 0
            else:
                caly = realx / cosalpha * sinalpha
                value = 1 if math.fabs(caly - realy) <= 2 else 0
            if value == 1:
                total += 1
            kernel[y,x] = value
    if total > 0:
        kernel = kernel / total
    kernel = np.flipud(kernel) # 翻转kernel为正
    return kernel

def testBlur(img):
    for ang in range(-360, 360, 15):
        kernel=blurKernel(length = 20, angle = ang)
        motion_blur=cv2.filter2D(img,-1,kernel)
        # blur = cv2.GaussianBlur()
        winname = 'test motion {}'.format(ang)
        cv2.imshow(winname, motion_blur)
        cv2.waitKey(0)
        cv2.destroyWindow(winname)

def flowWithFrames(frame1, frame2):
    # 必须转灰图
    prevF = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    nextF = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    # flow 为每一个像素点的偏移
    flow = cv2.calcOpticalFlowFarneback(prevF,nextF, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def imgAddBlur(src1, src2, mask):
    # shape = src1.shape
    res = src1.copy()
    mask3 = mask.astype(np.float)
    locations = np.where(mask3 > 0)

    val = mask3[locations]
    percent = val * 2 / 255.0
    percent[percent > 1] = 1
    percent = percent.reshape(-1, 1)
    po = src1[locations]
    pd = src2[locations]
    res[locations] = po * (1 - percent) + pd * percent
    return res

def motionBlurWithFrames(frame1, frame2):
    flow = flowWithFrames(frame1, frame2)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    test_round_x = 24
    test_round_y = test_round_x / 2
    test_width = int(frame1.shape[1] / test_round_x)
    test_height = int(frame1.shape[0] / test_round_y)
    # res = np.zeros_like(frame1)
    res = frame1.copy()
    emptymask = np.zeros_like(frame1)
    emptymask = cv2.cvtColor(emptymask, cv2.COLOR_BGR2GRAY)
    # mask = emptymask.copy()
    for in_x in range(test_round_x):
        for in_y in range(test_round_y):
            startx = in_x * test_width
            endx = startx + test_width
            starty = in_y * test_height
            endy = starty + test_height

            mag_roi_flat = mag[starty:endy, startx:endx].flatten()
            mag_max = int(mag_roi_flat.max())
            mag_using = int(np.average(mag_roi_flat[mag_roi_flat > (mag_max - 1)]))
            mag_using = mag_using / 2
            if mag_using < 1:
                continue

            ang_roi_flat = ang[starty:endy, startx:endx].flatten()
            ang_using = int(np.average(ang_roi_flat[mag_roi_flat > (mag_max - 1)]))
            kernel = blurKernel(length = mag_using, angle = ang_using)
            blur_ro = cv2.filter2D(frame1,-1,kernel)
            
            blurmask = emptymask.copy()
            blurmask[starty:endy, startx:endx, ...] = 255
            blurmask = cv2.filter2D(blurmask, -1, kernel)
            
            res = imgAddBlur(res, blur_ro, blurmask)
            print('rounds: {},{}'.format(in_x, in_y))
    return res

def blurVideo():
    print("start")
    input = 'resource/testtre.mp4'
    output = 'test.mp4'
    cap = cv2.VideoCapture(input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cacheVideoName = "cachevideo.mp4"
    cc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cacheVideoName, cc, fps, (width, height))
    lastframe = np.array([])
    currentframeindex = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        if lastframe.any():
            thisframe = motionBlurWithFrames(lastframe, frame)
            out.write(thisframe)
        else:
            out.write(frame)
        lastframe = frame
        print('progress:{}/{}'.format(currentframeindex, framecount))
        currentframeindex += 1
    cap.release()
    out.release()
    ffmpegcommand = "ffmpeg -i {} -i {} -map 0:v -map 1:a -c:v libx264 -c:a copy {}".format(cacheVideoName, input, output)
    print('command: {}'.format(ffmpegcommand))
    os.system(ffmpegcommand)
    print("cleaning cache")
    os.system("rm -rf {}".format(cacheVideoName))
    print("end")

def testImages():
    frame1 = cv2.imread('resource/one1.png')
    frame2 = cv2.imread('resource/one2.png')
    res = motionBlurWithFrames(frame1, frame2)
    cv2.imshow('blr', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    blurVideo()
    # testImages()