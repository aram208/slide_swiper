# -*- coding: utf-8 -*-
import cv2
import imutils

class MotionDetector:
    def __init__(self, accumWeight = 0.5):
        # The larger the accumWeight is, the less older frames are weighted 
        # when computing the running average of the background. 
        # The smaller the accumWeight  value, the more older frames 
        # contribute to the running average.
        self.accumWeight = accumWeight
        # background model
        self.bg = None 

    def update_background(self, image):
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def detect(self, image, tValue = 25):
        # compute the absolute difference between the background model and the image
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        threshold = cv2.threshold(delta, tValue, 255, cv2.THRESH_BINARY)[1]
        _, contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None

        return (threshold, max(contours, key = cv2.contourArea))