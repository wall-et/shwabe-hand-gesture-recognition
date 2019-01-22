# https://medium.com/@muehler.v/simple-hand-gesture-recognition-using-opencv-and-javascript-eb3d6ced28a0

import numpy as np
from numpy import linalg as LA
import cv2
import math

cap = cv2.VideoCapture(1)

Lower_hsv_blue = np.array([110, 50, 50])
Upper_hsv_blue = np.array([130, 255, 255])


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

    color_contours = (0, 255, 0)
    color_hull = (255, 0, 0)

    contours_mask = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), np.uint8)

    max_dist = 10
    filter_value = 50
    points = []

    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)

        hullIndices = []
        hullIndices.append(cv2.convexHull(max_contour, returnPoints=False))

        contourPoints = cv2.convexHull(max_contour, True)

        cv2.drawContours(contours_mask, [max_contour], -1, (0, 255, 255), 2)

        if len(hullIndices[0]) > 3:

            defects = cv2.convexityDefects(max_contour, hullIndices[0])

            if type(defects) is np.ndarray:
                fingers = 0

                # Get defect points and draw them in the original image
                if defects is not None:
                    # print('defects shape = ', defects.shape[0])
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])

                        cv2.circle(contours_mask, far, 8, [211, 84, 0], -1)
                        #  finger count
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                        area = cv2.contourArea(max_contour)

                        if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                            fingers += 1

                            cv2.circle(contours_mask, far, 5, [0, 0, 255], -1)

                        if len(max_contour) >= 5:
                            (x_centre, y_centre), (minor_axis, major_axis), angle_t = cv2.fitEllipse(max_contour)

                    print(fingers)
                    print("**********>>>***************")

    cv2.imshow("countours_mask", contours_mask)

    k = cv2.waitKey(30) & 0xFF
    if k == 32:
        break

cap.release()
cv2.destroyAllWindows()