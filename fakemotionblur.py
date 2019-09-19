# coding:utf-8
import numpy as np
from cv2 import cv2
import math
import os
import time
from tqdm import tqdm

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

maxCorners = 50000
color = np.random.randint(128,255,(maxCorners,3))

def flowLucasWithFrames(frame1, frame2):
    print("tracking")
    prevF = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    nextF = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    rows, cols, _ = frame1.shape
    somesize = int(cols / 30)
    trackingImage = prevF
    p0 = cv2.goodFeaturesToTrack(image = trackingImage, maxCorners = maxCorners, qualityLevel = 0.01, minDistance = 1, blockSize = 5)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prevImg = prevF, nextImg = nextF, prevPts = p0, nextPts = None, winSize = (somesize, somesize), maxLevel = 10)

    good_p0 = p0[st == 1]
    good_p0 = good_p0.astype(np.int)
    good_p1 = p1[st == 1]
    goodxy = good_p1-good_p0
    allxy = np.zeros((rows, cols, 2))
    allxy[good_p0[...,1], good_p0[...,0]] = goodxy
    allxy = allxy
    mag, ang = cv2.cartToPolar(allxy[...,0], allxy[...,1])
    ang = -ang * 180.0 / math.pi # 这原本是弧度！
    
    print("drawing")
    res = frame1.copy()
    # for i,(old,new) in enumerate(zip(good_p0, good_p1)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     res = cv2.line(res, (a,b),(c,d), color[i].tolist(), 2)
    #     # img = cv2.circle(img,(a,b),5,color[i].tolist(),-1)
    #     res = cv2.circle(res,(c,d),5,color[i].tolist(),-1)

    test_round_x = 24
    test_round_y = int(test_round_x / 2)
    test_width = int(frame1.shape[1] / test_round_x)
    test_height = int(frame1.shape[0] / test_round_y)
    # res = np.zeros_like(frame1)
    res = frame1.copy()
    rows, cols, _ = frame1.shape
    emptymask = np.zeros((rows, cols, 1))
    # mask = emptymask.copy()
    round_x_progress = tqdm(range(test_round_x))
    for in_x in round_x_progress:
        for in_y in range(test_round_y):
            startx = in_x * test_width
            endx = startx + test_width
            starty = in_y * test_height
            endy = starty + test_height

            mag_roi_flat = mag[starty:endy, startx:endx].flatten()
            mag_max = int(mag_roi_flat.max())
            mag_using = int(np.average(mag_roi_flat[mag_roi_flat > (mag_max - 1)]))
            if mag_using < 2:
                continue

            ang_roi_flat = ang[starty:endy, startx:endx].flatten()
            ang_using = int(np.average(ang_roi_flat[mag_roi_flat > (mag_max - 1)]))
            mag_using = mag_using / 2
            kernel = blurKernel(length = mag_using, angle = ang_using)
            blur_roi = cv2.filter2D(frame1,-1,kernel)
            
            blurmask = emptymask.copy()
            blurmask[starty:endy, startx:endx] = 1
            blurmask = cv2.filter2D(blurmask, -1, kernel)
            
            # draw the blur
            locations = np.where(blurmask > 0)
            val = blurmask[locations] * 3
            val[val > 1] = 1
            val = val.reshape(-1, 1)
            res[locations] = res[locations] * (1 - val) + blur_roi[locations] * val
            # print('rounds: {},{}'.format(in_x, in_y))
    # return res

    print("done")
    name = 'lk'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640, 480)
    cv2.imshow(name, res)
    cv2.waitKey(0)
    cv2.destroyWindow('lk')

    

def flowFarneWithFrames(frame1, frame2):
    # 必须转灰图
    prevF = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    nextF = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    # flow 为每一个像素点的偏移
    flow = cv2.calcOpticalFlowFarneback(prevF,nextF, None, 0.5, 3, 15, 10, 5, 1.2, 0)
    return flow

def motionBlurWithFrames(frame1, frame2):
    flow = flowFarneWithFrames(frame1, frame2)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    ang = ang * 180.0 / math.pi # 这原本是弧度！
    test_round_x = 100
    test_round_y = int(test_round_x / 2)
    test_width = int(frame1.shape[1] / test_round_x)
    test_height = int(frame1.shape[0] / test_round_y)
    # res = np.zeros_like(frame1)
    res = frame1.copy()
    rows, cols, _ = frame1.shape
    emptymask = np.zeros((rows, cols, 1))
    # mask = emptymask.copy()
    round_x_progress = tqdm(range(test_round_x))
    for in_x in round_x_progress:
        for in_y in range(test_round_y):
            startx = in_x * test_width
            endx = startx + test_width
            starty = in_y * test_height
            endy = starty + test_height

            mag_roi_flat = mag[starty:endy, startx:endx].flatten()
            mag_max = int(mag_roi_flat.max())
            mag_using = int(np.average(mag_roi_flat[mag_roi_flat > (mag_max - 1)]))
            if mag_using < 2:
                continue

            ang_roi_flat = ang[starty:endy, startx:endx].flatten()
            ang_using = int(np.average(ang_roi_flat[mag_roi_flat > (mag_max - 1)]))
            mag_using = mag_using / 2
            kernel = blurKernel(length = mag_using, angle = ang_using)
            blur_roi = cv2.filter2D(frame1,-1,kernel)
            
            blurmask = emptymask.copy()
            blurmask[starty:endy, startx:endx] = 1
            blurmask = cv2.filter2D(blurmask, -1, kernel)
            
            # draw the blur
            locations = np.where(blurmask > 0)
            val = blurmask[locations] * 3
            val[val > 1] = 1
            val = val.reshape(-1, 1)
            res[locations] = res[locations] * (1 - val) + blur_roi[locations] * val
            # print('rounds: {},{}'.format(in_x, in_y))
    return res

def blurVideo():
    print("start")
    timestring = time.strftime('%Y%m%d_%H%M%S')
    input = '/Users/jam/Desktop/test_car.mp4'
    output = '/Users/jam/Desktop/testres{}.mp4'.format(timestring)
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
    frame1 = cv2.imread('resource/test1.jpg')
    frame2 = cv2.imread('resource/test2.jpg')
    # res = motionBlurWithFrames(frame1, frame2)
    # cv2.imshow('blr', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # test
    imgsize = (1920, 1080)
    frame1 = cv2.resize(frame1, imgsize)
    frame2 = cv2.resize(frame2, imgsize)
    flowLucasWithFrames(frame1, frame2)

if __name__ == "__main__":
    # blurVideo()
    testImages()