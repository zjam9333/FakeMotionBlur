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
    while test_x < cols: #round_x_progress:
        # round_x_progress.set_description('roughly blur drawing')
        startx = test_x
        endx = startx + test_width if startx + test_width < cols else cols
        test_x = endx
        if startx == endx:
            break 
        test_y = 0

        while test_y < rows:
            if shouldprintblurprogress:
                print("test", test_x, test_y)
            starty = test_y
            endy = starty + test_height if starty + test_height < rows else rows
            test_y = endy
            if starty == endy:
                break

            flow_roi = allflow[starty:endy, startx:endx]
            flow_avg_0 = np.array((np.average(flow_roi[...,0])))
            flow_avg_1 = np.array((np.average(flow_roi[...,1])))
            mag, ang = cv2.cartToPolar(flow_avg_0, flow_avg_1)
            if mag < 2:
                continue
            mag = mag * scale
            ang = -ang * 180.0 / math.pi # 这原本是弧度！
            mag_using = mag[0,0]
            ang_using = ang[0,0]

            region = (starty, endy, startx, endx)
            drawRegionBlur(src=src, dsc=dsc, region=region, mag=mag_using, ang=ang_using, maskvalue=1.5)
    return dsc

def carefullyBlur(src, dsc, points, vectors):
    print('carefully blur drawing')
    rows, cols, _ = src.shape
    # zipedpoints = zip(points, vectors)
    rangepoints = tqdm(range(points.shape[0]))
    drawradius = 10
    for i in rangepoints:
        poi = points[i]
        vec = vectors[i]
        # print(poi, vec)
        mag, ang = cv2.cartToPolar(vec[0], vec[1])
        ang = -ang * 180.0 / math.pi # 这原本是弧度！
        mag_using = mag[0]
        ang_using = ang[0]

        ix = 0
        iy = 1
        startx = poi[ix] - drawradius
        endx = poi[ix] + drawradius
        starty = poi[iy] - drawradius
        endy = poi[iy] + drawradius
        if startx < 0:
            startx = 0
        if endx > cols - 1:
            endx = cols - 1
        if starty < 0:
            starty = 0
        if endy > rows - 1:
            endy = rows - 1

        region = (starty, endy, startx, endx)
        drawRegionBlur(src=src, dsc=dsc, region=region, mag=mag_using, ang=ang_using)
    return dsc

maxCorners = 50000
color = np.random.randint(128,255,(maxCorners,3))

def flowLucasWithFrames(frame1, frame2):
    print("tracking")
    prevF = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    nextF = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    rows, cols, _ = frame1.shape
    somesize = int(cols / 20)
    trackingImage = prevF
    p0 = cv2.goodFeaturesToTrack(image = trackingImage, maxCorners = maxCorners, qualityLevel = 0.01, minDistance = 1, blockSize = 5)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prevImg = prevF, nextImg = nextF, prevPts = p0, nextPts = None, winSize = (somesize, somesize), maxLevel = 10)

    good_p0 = p0[st == 1]
    good_p0 = good_p0.astype(np.int)
    good_p1 = p1[st == 1]
    goodxy = good_p1-good_p0
    allxy = np.zeros((rows, cols, 2))
    allxy[good_p0[...,1], good_p0[...,0]] = goodxy
    # allxy = allxy

    res = frame1.copy()
    for i,(old,new) in enumerate(zip(good_p0, good_p1)):
        a,b = new.ravel()
        c,d = old.ravel()
        res = cv2.line(res, (a,b),(c,d), color[i].tolist(), 1)
        # img = cv2.circle(img,(a,b),5,color[i].tolist(),-1)
        res = cv2.circle(res,(c,d),2,color[i].tolist(),-1)
    cv2.imshow('flow', res)
    cv2.waitKey(1)
    
    res = frame1.copy()
    res = roughlyBlur(src=frame1, dsc=res, allflow=allxy)
    res = carefullyBlur(src=frame1, dsc=res, points=good_p0, vectors=goodxy)
    return res

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

def simpleAddWithFrames(frame1, frame2):
    # hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    # hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    # val1 = hsv1[:,:,2]
    # val2 = hsv2[:,:,2]
    # where = np.where(val1 < val2)
    # hsv1[where] = hsv2[where]
    # return cv2.cvtColor(hsv1, cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(frame1, 0.5, frame2, 0.5, 1)

def testImages():
    frame1 = cv2.imread('resource/test1.jpg')
    frame2 = cv2.imread('resource/test2.jpg')
    # res = motionBlurWithFrames(frame1, frame2)
    # cv2.imshow('blr', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # test
    imgsize = (1280, 720)
    frame1 = cv2.resize(frame1, imgsize)
    frame2 = cv2.resize(frame2, imgsize)
    # res = flowLucasWithFrames(frame1, frame2)
    res = flowFarnebackWithFrames(frame1, frame2, fps = 60)
    # res = simpleAddWithFrames(frame1, frame2)

    name = 'blur'
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 640, 480)
    cv2.imshow(name, res)
    cv2.waitKey(0)
    cv2.destroyWindow(name)

if __name__ == "__main__":
    shouldprintblurprogress = True
    # blurVideo()
    testImages()
    # testBlur()