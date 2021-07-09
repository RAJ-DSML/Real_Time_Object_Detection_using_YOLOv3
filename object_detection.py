'''

Author Name   : RAJ KUMAR PAL
Task 1        : Object Detection
Domain        : Computer Vision & Internet of Things

GRIP @ The Sparks Foundation - JULY 2021
This is 2nd file <object_detection>

'''

import numpy as np
import cv2
import time


def show_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def labels_and_boxes(img, boxes, confidences, class_ids, idx_s, colors, labels):
    # If there are any detections
    if len(idx_s) > 0:
        for i in idx_s.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # Get the unique color for this class
            color = [int(c) for c in colors[class_ids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:4f}".format(labels[class_ids[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def generate_boxes_conf_cids(outs, height, width, tconf):
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:

            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:

                box = detection[0:4] * np.array([width, height, width, height])
                x_center, y_center, box_width, box_height = box.astype('int')

                # Using the center x and center y coordinates to derive the top and the left corner of the bounding box
                x = int(x_center - (box_width / 2))
                y = int(y_center - (box_height / 2))

                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(classid)

    return boxes, confidences, class_ids


def image_infer(net, layer_names, height, width, img, colors, labels, FLAGS,
                boxes=None, confidences=None, class_ids=None, idx_s=None, infer=True):
    if infer:

        # Blob from the input image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        if FLAGS.show_time:
            print("[INFO...] YOLOv3 took {:6f} seconds".format(end - start))

        # Generate the boxes, confidences and class_ids
        boxes, confidences, class_ids = generate_boxes_conf_cids(outs, height, width, FLAGS.confidence)

        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idx_s = cv2.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)

    if boxes is None or confidences is None or idx_s is None or class_ids is None:
        raise '[INFO...] Required variables are set to None before drawing boxes on images.'

    # Draw labels and boxes on the image
    img = labels_and_boxes(img, boxes, confidences, class_ids, idx_s, colors, labels)

    return img, boxes, confidences, class_ids, idx_s