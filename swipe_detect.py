from utils import MotionDetector
from utils import SwipeDetector
from utils.selenium.slidemanager import SlideManager
from face_detect import FaceDetector
import numpy as np
import argparse
import imutils
import cv2
import json

# ---------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-cfg", "--config", default = "conf.json", help = "Path to the config file (default one is conf.json)")
args = vars(ap.parse_args())
# ---------------------------

cfg = json.loads(open(args["config"]).read())

face_detector = FaceDetector(
    cfg["face_detect"]["detector"], 
    cfg["face_detect"]["embedding_model"], 
    cfg["face_detect"]["recognizer"],
    cfg["face_detect"]["le"],
    cfg["face_detect"]["confidence"])

camera = cv2.VideoCapture(0) # 0 for internal webcam, 1 for external

(top_h, right_h, bottom_h, left_h) = np.int32(cfg["bounding_box_hand"].split(","))
(top_f, right_f, bottom_f, left_f) = np.int32(cfg["bounding_box_face"].split(","))

motion_detector = MotionDetector()
swipe_detector = SwipeDetector()
slide_manager = SlideManager(cfg["chromedriver_path"])
slide_manager.navigate_to(cfg["slide_url"], cfg["slide_title"])

number_of_frames = 0
detection_in_progress = False
# X = []
x_start = 0
x_end = 0

presenters = cfg["face_detect"]["presenters"]

print("[INFO] Authorized presenters: {}".format(presenters))

while True:
    (is_success, frame) = camera.read()

    frame = imutils.resize(frame, width = 600)
    frame = cv2.flip(frame, 1)

    roi_hand = frame[top_h:bottom_h, right_h:left_h]
    roi_face = frame[top_f:bottom_f, right_f:left_f]

    if number_of_frames < 32:
        clone = frame.copy()
        (frameHeight, frameWidth) = frame.shape[:2]
        gray = cv2.cvtColor(roi_hand, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        motion_detector.update_background(gray)

    else:

        name, confidence, (startX, startY, endX, endY) = face_detector.detect_and_recognize(roi_face)
        text = "{}: {}".format(name, confidence)

        if name in presenters:            
            clone = frame.copy()

            cv2.rectangle(clone, (left_f, top_f), (right_f, bottom_f), (0, 255, 0), 2) # the box for face detection
            y = startY - 10 if startY - 10 > 10 else startY + 10
            #cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2) # the box around the face
            cv2.putText(clone, text, (right_f + 10, bottom_f - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

            (frameHeight, frameWidth) = frame.shape[:2]
            cv2.rectangle(clone, (left_h, top_h), (right_h, bottom_h), (0, 255, 0), 2) # the box for hand detection

            gray = cv2.cvtColor(roi_hand, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            hand = motion_detector.detect(gray, float(cfg["threshold"]))
            if hand is not None:
                detection_in_progress = True
                (thresh, c) = hand
                cv2.drawContours(clone, [c + (right_h, top_h)], -1, (0, 255, 0), 2)
                # cv2.imshow("Thresh", thresh)
                # print the coordinates of the center of mass of the contour
                (cX, cY) = swipe_detector.detect(thresh, c)
                x1 = cX + right_h
                y1 = cY + top_h
                cv2.circle(clone, (x1, y1), 10, (0, 255, 0), -1)
                
                #X.append(x1)
                
                if x_start == 0:
                    x_start = x1
                else:
                    x_end = x1

            else:
                if detection_in_progress:
                    # To determine the direction, 
                    # Option 1: compute the average X and compare it to the last X registered
                    # Option 2: just compare the first and last x coordinates
                    
                    # Option 1:
                    # average = sum(X) / len(X)
                    #print("min: {}, max: {}".format(min(X[:midpoint]), max(X[midpoint:])))
                    #if X[-1:] > average:
                    
                    # Option 2:
                    if x_start < x_end:
                        print("-> right swipe detected")
                        slide_manager.move_right()
                    else:
                        print("<- left swipe detected")
                        slide_manager.move_left()
                    
                    # reset the X (that is if using optoin 1 described above)
                    #X.clear()
                    x_start = 0
                    x_end = 0
                    detection_in_progress = False
        else:
            y = startY - 10 if startY - 10 > 10 else startY + 10
            #cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (right_f + 10, bottom_f - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            clone = frame.copy()
            cv2.rectangle(clone, (left_h, top_h), (right_h, bottom_h), (0, 0, 255), 2)
            cv2.rectangle(clone, (left_f, top_f), (right_f, bottom_f), (0, 0, 255), 2)


    #cv2.rectangle(clone, (left, top), (right, bottom), (0, 0, 255), 2)
    number_of_frames += 1

    cv2.imshow("Frame", clone)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
slide_manager.close_webdriver()
