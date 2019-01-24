# create morphology and detect it on image

import numpy as np
import cv2
from collections import deque
import os
import sys


# import argparse

class ImageProcessor(object):

    def __init__(self):

        self.pts = deque(maxlen=64)

        # self.lower_blue = np.array([110, 50, 50])
        # self.upper_blue = np.array([130, 255, 255])
        self.lower_blue = np.array([0, 45, 80])
        self.upper_blue = np.array([20, 198, 255])

    # self.lower_blue = np.array([0, 30, 60])
        # self.upper_blue = np.array([20, 150, 255])

    def resource_path(self,relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def remove_face(self, img):
        path = "haarcascade_frontalface_alt.xml"
        dpath = self.resource_path(path)
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        face_cascade = cv2.CascadeClassifier(dpath)
        col = (0, 0, 0)
        gray = cv2.equalizeHist(cv2.cvtColor(img,
                                                  cv2.COLOR_BGR2GRAY))

        detected = list(face_cascade.detectMultiScale(gray,
                                                           scaleFactor=1.3,
                                                           minNeighbors=4,
                                                           minSize=(
                                                               50, 50),
                                                           flags=cv2.CASCADE_SCALE_IMAGE))
        if len(detected) > 0:

            for face in detected:
                face_rect = face

                x, y, w, h = face_rect
                x = int(x*0.95)
                y = int(y*0.85)
                w = int(w*1.2)
                h = int(h*1.5)
                cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)

        cv2.imshow("img-----", img)
        return img

        self.img = None
        self.mask = None
        self.masked_image = None

    def extract_mask(self, img):

        self.img = self.remove_face(img)
        # self.img = cv2.flip(img, 1)
        # cv2.imshow("img", img)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        rangeMask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        # cv2.imshow("rangeMask", rangeMask)

        self.mask = cv2.blur(rangeMask, (10, 10))
        # cv2.imshow("blr", mask)
        ret, self.mask = cv2.threshold(self.mask, 150, 255, cv2.THRESH_BINARY)
        # cv2.imshow("blr", mask)

        kernel = np.ones((5, 5), np.uint8)
        self.mask = cv2.dilate(self.mask, kernel, iterations=2)
        # cv2.imshow("dilate", mask)

        self.masked_image = cv2.bitwise_and(self.img, self.img, mask=self.mask)

        return self.mask.copy()

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

    # def draw_circle(self):
    #
    #     if len(self.cnts) > 0:
    #
    #         c = max(self.cnts, key=cv2.contourArea)
    #         # find the circumcircle of an object
    #         ((x, y), radius) = cv2.minEnclosingCircle(c)
    #         # Calculates all of the moments up to the third order of a polygon or rasterized shape.
    #         M = cv2.moments(c)
    #         # Image moments help you to calculate some features like center of mass of the object, area of the object etc
    #         center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #
    #         if radius > 3:
    #             cv2.circle(self.img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    #             cv2.circle(self.img, center, 5, (0, 0, 255), -1)
    #
    #         self.pts.appendleft(center)
    #         return center
    #     return None
    #
    # def draw_line(self):
    #
    #     for i in range(1, len(self.pts)):
    #         # for i in xrange(1, len(pts)):
    #         if self.pts[i - 1] is None or self.pts[i] is None:
    #             continue
    #
    #         thick = int(np.sqrt(len(self.pts) / float(i + 1)) * 2.5)
    #         cv2.line(self.img, self.pts[i - 1], self.pts[i], (0, 0, 225), thick)

    def draw_windows(self):
        cv2.imshow("Frame", self.img)
        cv2.imshow("mask", self.mask)
        cv2.imshow("res", self.masked_image)

        k = cv2.waitKey(30) & 0xFF

        if k == 32:
            return




# image = ImageProcessor()
