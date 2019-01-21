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
    cv2.imshow("img", img)
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
        # max_contour.append(max(contours, key=cv2.contourArea))
        max_contour = max(contours, key=cv2.contourArea)
        hull.append(cv2.convexHull(max_contour, False))
        # defects = cv2.convexityDefects(max_contour, hull)
        # defects = cv2.convexityDefects(max_contour, hull)

        # cunvex_points = cv2.convexHull((max_contour, True))
        # print("------------------",contours[0])
        cv2.drawContours(contours_mask, [max_contour], -1, (0, 255, 255), 2)
        print("==================",hull[0])
        for index in range(len(hull[0])):
            cv2.circle(contours_mask, (hull[0][index][0][0], hull[0][index][0][1]), 3, (0, 0, 255), -1)

        # for index in range(len(defects[0])):
        #     cv2.circle(contours_mask, (defects[0][index][0][0], defects[0][index][0][1]), 3, (0, 0, 255), -1)
            # cv2.drawContours(contours_mask, (hull[0][0][0][0],hull[0][0][0][1]), i, color_contours, 2, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(contours_mask, hull, 0, color_hull, 1, 8)

        points_vec = np.float32(hull[0])

        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        #
        # # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_RANDOM_CENTERS
        #
        # # Apply KMeans
        # compactness, labels, centers = cv2.kmeans(points_vec, 7, None, criteria, 10, flags)
        clusters_amount = 7
        if len(points_vec) > clusters_amount:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(points_vec, clusters_amount, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # print("labels-------",label)
            A = points_vec[label.ravel() == 0]
            B = points_vec[label.ravel() == 1]
            C = points_vec[label.ravel() == 2]
            D = points_vec[label.ravel() == 3]
            E = points_vec[label.ravel() == 4]
            F = points_vec[label.ravel() == 5]
            G = points_vec[label.ravel() == 6]
            # print("=================",A[0][0][0])
            cv2.circle(contours_mask, (A[0][0][0], A[0][0][1]), 3, (0, 255, 0), -1)
            cv2.circle(contours_mask, (B[0][0][0], B[0][0][1]), 3, (0, 255, 0), -1)
            cv2.circle(contours_mask, (C[0][0][0], C[0][0][1]), 3, (0, 255, 0), -1)
            cv2.circle(contours_mask, (D[0][0][0], D[0][0][1]), 3, (0, 255, 0), -1)
            cv2.circle(contours_mask, (E[0][0][0], E[0][0][1]), 3, (0, 255, 0), -1)
            cv2.circle(contours_mask, (F[0][0][0], F[0][0][1]), 3, (0, 255, 0), -1)
            cv2.circle(contours_mask, (G[0][0][0], G[0][0][1]), 3, (0, 255, 0), -1)
        # # extLeft = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
        # extRight = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
        # extTop = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
        # extBot = tuple(max_contour[max_contour[:, :, 1].argmax()][0])
        #
        # cv2.circle(contours_mask, extLeft, 8, (0, 0, 255), -1)
        # cv2.circle(contours_mask, extRight, 8, (0, 255, 0), -1)
        # cv2.circle(contours_mask, extTop, 8, (255, 0, 0), -1)
        # cv2.circle(contours_mask, extBot, 8, (255, 255, 0), -1)

    cv2.imshow("countours_mask", contours_mask)
    # if len(cnts) > 0:
    #     contour = max(cnts, key=cv2.contourArea)
    #
    #     hull = []
    #     hull.append(cv2.convexHull(contour, False))
    #     cv2.drawContours(drawing, list(contour),0, color_contours, 1, 8, heir)
    #     # draw ith convex hull object
    #     cv2.drawContours(drawing, hull, 0, color, 1, 8)
    #     # hullIndices = cv2.convexHull((contour,True))
    #     # hullIndices = contour.convexHullIndices()
    #     # contourPoints = contour.getPoints()
    #
    #     cv2.imshow("countours",drawing)

    # find the circumcircle of an object
    # ((x, y), radius) = cv2.minEnclosingCircle(c)
    # Calculates all of the moments up to the third order of a polygon or rasterized shape.
    # M = cv2.moments(c)
    # Image moments help you to calculate some features like center of mass of the object, area of the object etc
    # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #
    # if radius > 3:
    #     cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    #     cv2.circle(img, center, 5, (0, 0, 255), -1)

    # pts.appendleft(center)
    # cv2.imshow("img with circle", img)

    k = cv2.waitKey(30) & 0xFF
    if k == 32:
        break

cap.release()
cv2.destroyAllWindows()