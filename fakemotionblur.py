import numpy as np
from cv2 import cv2
import math

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

def blurKernel(length,angle):
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

def showFlow(flow):
    # 转为极坐标，mag极径，ang极角
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    hsv = np.array(np.zeros_like(frame1))
    hsv[...,1] = 255
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # vvv = hsv[...,2]

    rgb = hsv.copy() #cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # newvv = rgb[...,2]
    # thre = 0
    # newvv[vvv > thre] = 255
    # rgb[vvv <= thre] = 0

    rgb = cv2.cvtColor(rgb, cv2.COLOR_HSV2BGR)

    cv2.imshow("flow", rgb)

def imgAdd(src1, src2, mask):
    rows, cols, _ = src1.shape
    res = src1.copy()

    # 框取有需要的范围
    startrows = 0
    endrows = rows - 1
    for r in range(0,rows):
        if startrows >= endrows:
            break
        row0 = mask[r, ...]
        if row0.max() < 10:
            startrows = r
        row1 = mask[rows - 1 - r, ...]
        if row1.max() < 10:
            endrows = rows - r
    if startrows > endrows:
        return res

    for r in range(startrows, endrows):
        for c in range(cols):
            val = mask[r, c]
            if val == 0:
                continue
            percent = val / 255
            po = src1[r, c]
            pd = src2[r,c]
            res[r, c] = po * (1 - percent) + pd * percent
    return res

def motionBlurWithFrames(frame1, frame2):
    flow = flowWithFrames(frame1, frame2)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    test_round = 50
    test_width = int(frame1.shape[1] / test_round)
    test_height = int(frame1.shape[0] / test_round)
    res = np.zeros_like(frame1)
    # masksize = (frame1.shape[0], frame1.shape[1], 1)
    emptymask = np.zeros_like(frame1)
    emptymask = cv2.cvtColor(emptymask, cv2.COLOR_BGR2GRAY)
    mask = emptymask.copy()
    for in_x in range(test_round):
        for in_y in range(test_round):
            print('rounds:', in_x, in_y)
            startx = in_x * test_width
            endx = startx + test_width
            starty = in_y * test_height
            endy = starty + test_height
            mag_roi = mag[starty:endy, startx:endx]
            ang_roi = ang[starty:endy, startx:endx]
            if mag_roi.shape[0] == 0 or mag_roi.shape[1] == 0:
                continue
            mag_using = int(mag_roi.max())
            if mag_using < 10:
                continue
            ang_using = int(np.average(ang_roi[mag_roi > 10].flatten()))
            # # ang_using = int(-ang_using / np.pi * 180)
            # mag_using = 20
            # ang_using = 45
            kernel = blurKernel(length = mag_using, angle = ang_using)
            blur_ro = frame1.copy()
            # blur_ro[starty:endy, startx:endx] = frame1[starty:endy, startx:endx]
            blur_ro = cv2.filter2D(blur_ro,-1,kernel)

            # res[starty:endy, startx:endx] = blur_roi
            blurmask = emptymask.copy()
            blurmask[starty:endy, startx:endx, ...] = 255
            blurmask = cv2.filter2D(blurmask, -1, kernel)
            # cv2.imshow('blr', blurmask)
            # cv2.waitKey(0)
            # mask = cv2.bitwise_or(mask, blurmask)
            # mask = cv2.add(mask, blurmask)

            res = imgAdd(res, blur_ro, blurmask)
    
    # mask = np.zeros_like(mag)
    # mask[mag > 1] = 1
    # res = cv2.GaussianBlur(res, (5, 5), 0)
    # mask[mask > 255] = 255
    # return mask
    return res



if __name__ == "__main__":
    frame1 = cv2.imread('resource/one1.png')
    frame2 = cv2.imread('resource/one2.png')
    # size = (200,150)
    # frame1 = cv2.resize(frame1,size)
    # frame2 = cv2.resize(frame2,size)
    # flow = flowWithFrames(frame1,frame2)

    # showFlow(flow)
    # testBlur(frame1)
    res = motionBlurWithFrames(frame1, frame2)
    cv2.imshow('blr', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()