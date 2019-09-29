# coding:utf-8
'''
import numpy as np
from cv2 import cv2
import math
import os
import time
from tqdm import tqdm

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

def simpleAddWithFrames(frame1, frame2):
    # hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    # hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    # val1 = hsv1[:,:,2]
    # val2 = hsv2[:,:,2]
    # where = np.where(val1 < val2)
    # hsv1[where] = hsv2[where]
    # return cv2.cvtColor(hsv1, cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(frame1, 0.5, frame2, 0.5, 1)
'''