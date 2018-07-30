#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from src.yolo import YOLO
import colorsys
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
import argparse
from src.demo_image import POSE
warnings.filterwarnings('ignore')


def arg_parse():

    parser = argparse.ArgumentParser(description='openpose Video Detection Module')

    parser.add_argument("--video", dest = 'video', help = "Video to run detection upon",
                default = "/home/ftp/DataSet/Hand-Dataset/video/tennis.mp4", type = str)
    parser.add_argument('--model', dest = 'model', type=str, default='model/keras/model.h5', help='path to the weights file')
    return parser.parse_args()


def deep_sort():
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    # deep_sort 
    model_filename = 'model/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    return tracker, encoder, nms_max_overlap


def main(yolo):

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / 80, 1., 1.)
                for x in range(80)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    tracker, encoder, nms_max_overlap = deep_sort()
    classes = load_classes('model/coco_classes.txt')
    writeVideo_flag = True 
    args = arg_parse()
    videofile = args.video
    keras_weights_file = args.model
    # 加载openpose模型
    pose = POSE(keras_weights_file)
    video_capture = cv2.VideoCapture(videofile)
    assert video_capture.isOpened(), 'Cannot capture source'
    
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        outname = os.path.join('result_video',videofile.split('/')[-1])
        out = cv2.VideoWriter(outname, fourcc, 25, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    #athlete_flag = [0,0]

    while video_capture.isOpened():
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        image = Image.fromarray(frame)
        boxs = yolo.detect_image_track(image)
        boxs = np.array(boxs)
        #print("box_num",len(boxs))
        print("box_des",boxs)
        #cls = int(boxs[-1])
        #label = "{0}".format(classes[cls])
        #t_size = cv2.getTextSize(boxs[0][-1], cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox[:-1],bbox[-1], 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        flag_boxs = 0
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue 
            bbox = track.to_tlbr()
            label = "{0}".format(classes[int(track.nameclass)])
            c1 = (int(bbox[0]),int(bbox[1]))
            c2 = (int(bbox[2]),int(bbox[3]))
            if label == 'person':
                if c1[0]>256 and c1[1]>95 and c2[0]<1053 and c2[1]<717:
                    label = 'athlete'
                elif c1[0]>261 and c1[1]>20 and c2[0]<1012 and c2[1]<162:
                    label = 'baseline_referee'
                elif c1[0]>1 and c1[1]>95 and c2[0]<175 and c2[1]<413:
                    label = 'sideline_referee'
                elif c1[0]>1002 and c1[1]>128 and c2[0]<1181 and c2[1]<387:
                    label = 'referee'
                elif c1[0]>1002 and c1[1]>245 and c2[0]<1170 and c2[1]<387:
                    label = 'Caddy'
                elif c1[0]>166 and c1[1]>232 and c2[0]<295 and c2[1]<347:
                    label = 'Caddy'
            color = colors[int(track.nameclass)]
            cv2.rectangle(frame, c1, c2,color, 2)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 2)[0]
            name = label +'_' +str(track.track_id)
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(frame, c1, c2,color, -1)
            cv2.putText(frame, name, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0), 1);
            #cv2.putText(frame, str(track.nameclass)+'_'+str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            
            #cv2.putText(frame, str(boxs[flag_boxs][-1])+'_'+str(track.track_id),(int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
            flag_boxs += 1
            #canvas = pose.detect_pose(frame[37:704,69:1153,:])
            #frame[37:704,69:1153,:] = canvas
            
        cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

if __name__ == '__main__':
    main(YOLO())
