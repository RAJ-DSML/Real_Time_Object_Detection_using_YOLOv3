'''

Author Name   : RAJ KUMAR PAL
Task 1        : Object Detection
Domain        : Computer Vision & Internet of Things

GRIP @ The Sparks Foundation - JULY 2021
This is main file <main_detection>

'''

import numpy as np
import argparse
import cv2
import subprocess
from object_detection import image_infer

count = 0
FLAGS = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path', type=str, default='./yolo_supporting_files/')
    parser.add_argument('-w', '--weights', type=str, default='./yolo_supporting_files/yolov3.weights')
    parser.add_argument('-cfg', '--config', type=str, default='./yolo_supporting_files/yolov3.cfg')
    parser.add_argument('-l', '--labels', type=str, default='./yolo_supporting_files/coco.names')
    parser.add_argument('-c', '--confidence', type=float, default=0.5)
    parser.add_argument('-th', '--threshold', type=float, default=0.3)
    parser.add_argument('--download-model', type=bool, default=False)

    parser.add_argument('-t', '--show-time', type=bool, default=False)

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.download_model:
        subprocess.call(['./yolo_supporting_files/get_model.sh'])

    labels = open(FLAGS.labels).read().strip().split('\n')

    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    net = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    print('[INFO...] Starting the Webcam')

    vid = cv2.VideoCapture(1)
    while True:
        _, frame = vid.read()
        height, width = frame.shape[:2]

        if count == 0:
            frame, boxes, confidences, class_ids, idx_s = image_infer(net, layer_names, height, width, frame,
                                                                      colors, labels, FLAGS)
            count += 1
        else:
            frame, boxes, confidences, class_ids, idx_s = image_infer(net, layer_names, height, width, frame,
                                                                      colors, labels, FLAGS, boxes, confidences,
                                                                      class_ids, idx_s, infer=False)
            count = (count + 1) % 6

        cv2.imshow('LIVE EXTERNAL CAMERA FEED', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()