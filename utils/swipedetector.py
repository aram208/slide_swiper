# -*- coding: utf-8 -*-
import cv2

class SwipeDetector:
    def __init__(self):
        pass

    def detect (self, threshold, contour):
        
        hull = cv2.convexHull(contour)
        extremeLeft     = tuple(hull[hull[:, :, 0].argmin()][0])
        extremeRight    = tuple(hull[hull[:, :, 0].argmax()][0])
        extremeTop      = tuple(hull[hull[:, :, 1].argmin()][0])
        extremeBottom   = tuple(hull[hull[:, :, 1].argmax()][0])

        # compute the center (x, y) coordinates based on the extreme points
        cX = (extremeLeft[0] + extremeRight[0]) // 2
        cY = (extremeBottom[1] + extremeTop[1]) // 2
        cY = int(cY)

        # return the centroid
        return (cX, cY)

    @staticmethod
    def drawBox(roi, i, color=(0, 0, 255)):
        cv2.rectangle(roi, ((i * 50) + 10, 10), ((i * 50) + 50, 60), color, 2)