# https://medium.com/@muehler.v/simple-hand-gesture-recognition-using-opencv-and-javascript-eb3d6ced28a0

import numpy as np
import cv2
import math
import argparse
from collections import deque
from pynput.mouse import Controller
import scipy.cluster.hierarchy as hcluster

cap = cv2.VideoCapture(0)

Lower_hsv_blue = np.array([110, 50, 50])
Upper_hsv_blue = np.array([130, 255, 255])

# pts = deque(maxlen=64)
def filter_points(points, filter_value):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i] and points[j] and dist(points[i], points[j]) < filter_value:
                points[j] = None
    filtered = []
    for point in points:
        if point is not None:
            filtered.append(point)
    return filtered


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (b[1] - a[1]) ** 2)

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

    # hull = []
    max_dist = 10
    filter_value = 50
    points_tips = []
    points_ins = []

    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)

        # const hullIndices = contour.convexHullIndices();
        hullIndices = []
        hullIndices = cv2.convexHull(max_contour, returnPoints=False)
        cv2.drawContours(contours_mask, [max_contour], -1, (0, 255, 255), 2)
        cv2.drawContours(contours_mask, hullIndices, color_hull, 1, 8)
        print(len(hullIndices))

        # drawing hullindices
        # for index in range(len(hullIndices[0])):
        #     cv2.circle(contours_mask, (hullIndices[0][index][0][0], hullIndices[0][index][0][1]), 3, (0, 0, 255), -1)
        # for index in range(len(hullIndices[0])):
        #     cv2.circle(contours_mask, (hullIndices[0][index][0][0], hullIndices[0][index][0][1]), 3, (0, 0, 255), -1)

        # const contourPoints = contour.getPoints();
        # contourPoints = cv2.convexHull(max_contour, True)
        # print(len(contourPoints))
        # drawing contourPoints

        # for index in range(len(contourPoints)):
        #     cv2.circle(contours_mask, (contourPoints[index][0][0], contourPoints[index][0][1]), 3, (255, 0, 0), -1)
        # print(len(hullIndices[0]))
        # cv2.drawContours(contours_mask, [max_contour], -1, (0, 255, 255), 2)

        # for index in range(len(hullIndices[0])):
        #     print("====",hullIndices)
            # cv2.circle(contours_mask, (hullIndices[0][index][0], hullIndices[0][index][1]), 3, (0, 0, 255), -1)

    # cv2.drawContours(contours_mask, hullIndices, 0, color_hull, 1, 8)
        # for index in range(len(hullIndices[0])):
        #     cv2.circle(contours_mask, (hullIndices[0][index][0][0], hullIndices[0][index][0][1]), 3, (0, 0, 255), -1)

        # if len(hullIndices[0]) > 3:
        #     # print(type(hullIndices[0][0][0]))
        #     # print(hullIndices[0][0][0])
        #     # print(hullIndices[0].shape)
        #     defects = cv2.convexityDefects(max_contour, hullIndices[0])
        #     # print(type(defects))
        #     if type(defects) is np.ndarray:
        #         # if type(defects) == "<class 'numpy.ndarray'>":
        #         # print("==============================================")
        #         for i in range(defects.shape[0]):
        #             s, e, f, d = defects[i, 0]
        #             end = tuple(max_contour[e][0])
        #             start = tuple(max_contour[s][0])
        #             far = tuple(max_contour[f][0])
        #             points_tips.append(end)
        #             # points_tips.append(start)
        #             # points_ins.append(far)
        #         filtered = filter_points(points_tips, filter_value)
        #         filtere_far = filter_points(points_ins, filter_value)
        #         # print("points found",filtered)
        #         for index in range(len(filtered)):
        #             cv2.circle(contours_mask, (filtered[index][0], filtered[index][1]), 3, (255, 0, 0), -1)
        #
        #         for index in range(len(filtere_far)):
        #             cv2.circle(contours_mask, (filtere_far[index][0], filtere_far[index][1]), 3, (0, 0, 255), -1)


        # for i,cnt in enumerate(contours):
        # for index in range(len(contours)):
        #     hull = cv2.convexHull(contours[index], False)
        #     defects = cv2.convexityDefects(contours[index], hull)



        # hull.append(cv2.convexHull(max_contour, False))
        #
        # # draw contour

        # # print("==================", hull[0])
       #
        # # draw ith convex hull object


    cv2.imshow("countours_mask", contours_mask)

    k = cv2.waitKey(30) & 0xFF
    if k == 32:
        break

cap.release()
cv2.destroyAllWindows()
