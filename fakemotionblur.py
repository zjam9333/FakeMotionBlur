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

if __name__ == "__main__":
    frame1 = cv2.imread('resource/one1.png')
    # frame2 = cv2.imread('resource/one2.png')
    # flow = flowWithFrames(frame1,frame2)

    # showFlow(flow)
    testBlur(frame1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()