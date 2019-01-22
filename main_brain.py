
# get image data and create move from it
import math
import cv2
import numpy as np


class MainBrain:

    def __init__(self,m):
        self.mask = None
        self.calculated_mask = None
        self.defects = None
        self.mouse = m


    def find_contours(self, mask):

        self.mask = mask
        self.calculated_mask = np.zeros((self.mask.shape[0], self.mask.shape[1], 3), np.uint8)
        contours, hierarchy = cv2.findContours(self.mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            self.hand_contour = max_contour

            hullIndices = []
            hullIndices.append(cv2.convexHull(self.hand_contour, returnPoints=False))

            contourPoints = cv2.convexHull(self.hand_contour, True)

            # for debugging
            cv2.drawContours(self.calculated_mask, [self.hand_contour], -1, (0, 255, 255), 2)

            if len(hullIndices[0]) > 3:
                self.defects = cv2.convexityDefects(self.hand_contour, hullIndices[0])

    def find_defecets_point(self):

        if type(self.defects) is np.ndarray:
            fingers = 0
            far = (0,0)

            # Get defect points and draw them in the original image
            if self.defects is not None:
                # print('defects shape = ', defects.shape[0])
                for i in range(self.defects.shape[0]):
                    s, e, f, d = self.defects[i, 0]
                    start = tuple(self.hand_contour[s][0])
                    end = tuple(self.hand_contour[e][0])
                    far = tuple(self.hand_contour[f][0])

                    cv2.circle(self.calculated_mask, far, 8, [211, 84, 0], -1)
                    #  finger count
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    area = cv2.contourArea(self.hand_contour)

                    if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                        fingers += 1

                        cv2.circle(self.calculated_mask, far, 5, [0, 0, 255], -1)


                print(fingers)
                if fingers == 0:
                    self.mouse.release_all()
                if fingers == 1:
                    self.mouse.move(far)
                if fingers == 4:
                    self.mouse.left_click()
                print("**********>>>***************")

    def show_windows(self):

        cv2.imshow("calculated_mask", self.calculated_mask)