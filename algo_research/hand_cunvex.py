# https://medium.com/@muehler.v/simple-hand-gesture-recognition-using-opencv-and-javascript-eb3d6ced28a0

import numpy as np
import cv2
import argparse
from collections import deque
from pynput.mouse import Controller


cap = cv2.VideoCapture(0)

Lower_hsv_blue = np.array([110, 50, 50])
Upper_hsv_blue = np.array([130, 255, 255])

# pts = deque(maxlen=64)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    # cv2.imshow("img", img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    rangeMask = cv2.inRange(hsv, Lower_hsv_blue, Upper_hsv_blue)
    # cv2.imshow("rangeMask", rangeMask)

    mask = cv2.blur(rangeMask, (10, 10))
    # cv2.imshow("blr", mask)
    ret, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
    # cv2.imshow("blr", mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # cv2.imshow("dilate", mask)

    masked_image = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("mask", mask)
    # cv2.imshow("masked image", masked_image)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color_contours = (0, 255, 0)
    color_hull = (255, 0, 0)

    contours_mask = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), np.uint8)

    hull = []

    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        hull.append(cv2.convexHull(max_contour, False))

        # draw contour
        cv2.drawContours(contours_mask, [max_contour], -1, (0, 255, 255), 2)
        # print("==================", hull[0])
        for index in range(len(hull[0])):
            cv2.circle(contours_mask, (hull[0][index][0][0], hull[0][index][0][1]), 3, (0, 0, 255), -1)

        # draw ith convex hull object
        cv2.drawContours(contours_mask, hull, 0, color_hull, 1, 8)

    cv2.imshow("countours_mask", contours_mask)

    k = cv2.waitKey(30) & 0xFF
    if k == 32:
        break

cap.release()
cv2.destroyAllWindows()
