# https://medium.com/@muehler.v/simple-hand-gesture-recognition-using-opencv-and-javascript-eb3d6ced28a0

import numpy as np
from numpy import linalg as LA
import cv2
import math


cap = cv2.VideoCapture(1)

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

        # print(hullIndices[0])
        cv2.drawContours(contours_mask, [max_contour], -1, (0, 255, 255), 2)

        if len(hullIndices[0]) > 3:

            # print(hullIndices[0].shape)
            defects = cv2.convexityDefects(max_contour, hullIndices[0])
            # print(type(defects))
            if type(defects) is np.ndarray:

                # for i in range(defects.shape[0]):
                #     s, e, f, d = defects[i, 0]
                #     end = tuple(max_contour[e][0])
                #     far = tuple(max_contour[f][0])
                #     points.append(end)
                # filtered = filter_points(points, filter_value)
                # print("points found",filtered)
                # for index in range(len(filtered)):
                #     cv2.circle(contours_mask, (filtered[index][0], filtered[index][1]), 3, (255, 0, 0), -1)

                  # // get neighbor defect points of each hull point
                hullPointDefectNeighbors = {}

                for idx in hullIndices[0]:
                    # print(hullIndices[0][idx][0])
                    ind = idx[0]
                    hullPointDefectNeighbors[ind] = []

                for index in range(defects.shape[0]):
                    s, e, f, d = defects[index, 0]
                    startPointIdx = s
                    endPointIdx = e
                    defectPointIdx = f

                    # hullPointDefectNeighbors.get(startPointIdx).push(defectPointIdx);
                    hullPointDefectNeighbors[startPointIdx].append(defectPointIdx)

                    # hullPointDefectNeighbors.get(endPointIdx).push(defectPointIdx);
                    hullPointDefectNeighbors[endPointIdx].append(defectPointIdx)

                    getHullDefectVertices = hullPointDefectNeighbors.keys()

                    # (hullIndex = > hullPointDefectNeighbors.get(hullIndex).length > 1)

                    filtered_dict = {k: v for (k, v) in hullPointDefectNeighbors.items() if len(v) > 1}

                    defect_vertices = {}
                    # print("contourPoints",len(contourPoints))
                    # print(filtered_dict.keys())
                    for k in filtered_dict.keys():
                        defect_vertices[k] = {

                            'pt': max_contour[k],
                            'd1': max_contour[filtered_dict[k][0]],
                            'd2': max_contour[filtered_dict[k][1]]
                        }
                    # print("defect_vertices-------",len(defect_vertices))
                    # print("defect_vertices-------",defect_vertices)
                    if len(defect_vertices) > 0:
                        vertices_with_valid_angle = []
                        max_angle = 60
                        for k,v in defect_vertices.items():
                            # a = LA.norm(v['d1'] - v['d2'])
                            a = math.sqrt((v['d2'][0][0] - v['d1'][0][0])**2 + (v['d2'][0][1] - v['d1'][0][1])**2)
                            # b = LA.norm(v['pt'] - v['d1'])
                            b = math.sqrt((v['pt'][0][0] - v['d1'][0][0])**2 + (v['pt'][0][1] - v['d1'][0][1])**2)
                            # c = LA.norm(v['pt'] - v['d2'])
                            c = math.sqrt((v['d2'][0][0] - v['pt'][0][0])**2 + (v['d2'][0][1] - v['pt'][0][1])**2)

                            # print(f"a:{a},b:{b},c:{c}")
                            angleDeg = math.acos(((b*b + c*c) - a*a) / (2 * b * c))# * (180 / math.pi)
                            # print("==========",angleDeg)
                            cv2.circle(contours_mask, (v['pt'][0][0], v['pt'][0][1]), 3, (255, 0, 255), -1)
                            cv2.circle(contours_mask, (v['d1'][0][0], v['d1'][0][1]), 3, (100, 0, 255), -1)
                            cv2.circle(contours_mask, (v['d2'][0][0], v['d2'][0][1]), 3, (255, 0, 100), -1)
                            if angleDeg < math.pi / 2:
                                vertices_with_valid_angle.append(v['pt'][0])


                        print("!!!!!!!!!!!!!!!",len(vertices_with_valid_angle))
                        # for point in vertices_with_valid_angle:
                        #     cv2.circle(contours_mask, (point[0], point[1]), 3, (255, 0, 0), -1)




    cv2.imshow("countours_mask", contours_mask)

    k = cv2.waitKey(30) & 0xFF
    if k == 32:
        break

cap.release()
cv2.destroyAllWindows()