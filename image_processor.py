# create morphology and detect it on image

import numpy as np
import cv2
from collections import deque


# import argparse

class ImageProcessor(object):

    def __init__(self):

        self.pts = deque(maxlen=64)

        self.lower_blue = np.array([110, 50, 50])
        self.upper_blue = np.array([130, 255, 255])

    def extract_morph_from_img(self, img):

        self.img = cv2.flip(img, 1)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.kernel = np.ones((10, 10), np.uint8)  ###########
        self.mask = cv2.inRange(self.hsv, self.lower_blue, self.upper_blue)
        self.mask = cv2.dilate(self.mask, self.kernel, iterations=1)
        self.mask = cv2.erode(self.mask, self.kernel, iterations=2)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, self.kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, self.kernel)
        self.mask = cv2.dilate(self.mask, self.kernel, iterations=1)
        self.res = cv2.bitwise_and(self.img, self.img, mask=self.mask)
        self.cnts, self.heir = cv2.findContours(self.mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        self.center = None

    def draw_circle(self):

        if len(self.cnts) > 0:

            c = max(self.cnts, key=cv2.contourArea)
            # find the circumcircle of an object
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # Calculates all of the moments up to the third order of a polygon or rasterized shape.
            M = cv2.moments(c)
            # Image moments help you to calculate some features like center of mass of the object, area of the object etc
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 3:
                cv2.circle(self.img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(self.img, center, 5, (0, 0, 255), -1)

            self.pts.appendleft(center)
            return center
        return None

    def draw_line(self):

        for i in range(1, len(self.pts)):
            # for i in xrange(1, len(pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

            thick = int(np.sqrt(len(self.pts) / float(i + 1)) * 2.5)
            cv2.line(self.img, self.pts[i - 1], self.pts[i], (0, 0, 225), thick)

    def draw_windows(self):
        cv2.imshow("Frame", self.img)
        cv2.imshow("mask", self.mask)
        cv2.imshow("res", self.res)




image = ImageProcessor()
