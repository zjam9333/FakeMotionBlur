# coding:utf-8
import numpy as np
from cv2 import cv2
import os
import time
import fakemotionblur
from tqdm import tqdm

def handlevideo(inputpath, outputpath):
    print("start")
    input = '/Users/zjj/Downloads/out8.mp4'
    
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
    rangeframecount = tqdm(range(int(framecount)))
    for _ in rangeframecount:
        rangeframecount.set_description('frame progressing')
        ret, frame = cap.read()
        if ret == False:
            break
        if lastframe.any():
            # using a blur method from other module
            thisframe = fakemotionblur.flowFarnebackWithFrames(lastframe, frame, fps)
            out.write(thisframe)
        else:
            out.write(frame)
        lastframe = frame
        # print('progress:{}/{}'.format(currentframeindex, framecount))
        currentframeindex += 1
    cap.release()
    out.release()

    timestring = time.strftime('%Y%m%d_%H%M%S')
    output = '/Users/zjj/Downloads/out_crop_{}.mp4'.format(timestring)
    ffmpegcommand = "ffmpeg -i {} -i {} -map 0:v -map 1:a -c:v libx264 -c:a copy -r 30 {}".format(cacheVideoName, input, output)
    print('command: {}'.format(ffmpegcommand))
    os.system(ffmpegcommand)
    print("cleaning cache")
    os.system("rm -rf {}".format(cacheVideoName))
    print("end")

if __name__ == "__main__":
    # print('hello')
    handlevideo(None, None)