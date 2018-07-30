# -*- coding: utf-8 -*-
import argparse
import cv2
import os
import time
import numpy as np
from src.demo_image import POSE

def arg_parse():
   
    parser = argparse.ArgumentParser(description='openpose Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument('--model', dest = 'model', type=str, default='model/keras/model.h5', help='path to the weights file')
    return parser.parse_args()



if __name__ == '__main__':
    args = arg_parse()
    videofile = args.video
    keras_weights_file = args.model
    # 加载openpose模型
    pose = POSE(keras_weights_file)
    #_ = pose.detect_pose(np.ones((320,320,3)))
    cap = cv2.VideoCapture(videofile)
    assert cap.isOpened(), 'Cannot capture source'
    w = int(cap.get(3))
    h = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    outname = os.path.join('result_video2',videofile.split('/')[-1]) 
    out = cv2.VideoWriter(outname, fourcc, 25, (w, h))
    frames = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
       # canvas = pose.detect_pose(frame[40:700,80:1100,:])
        canvas = pose.detect_pose(frame[375:697,290:1100,:])
        #frame[40:700,80:1100,:] = canvas
        frame[375:697,290:1100,:] = canvas
        # Display the resulting frame
        cv2.imshow('Video', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frames += 1
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    cap.release()
    cv2.destroyAllWindows()
