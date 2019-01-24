# get image data and create move from it
import math
import cv2
import numpy as np


class MainBrain:

    def __init__(self):
        self.mask = None
        self.calculated_mask = None
        self.defects = None
        self.hand_contour = None
        # self.mouse = m
        self.move_cap = 2
        self.move_stats = [self.move_cap, 0, 0, 0, 0]
        self.last_center = None
        self.mouse_move_index = 1
        self.movement_delta = [0, 0]
        self.movement_thresh_x = 5
        self.movement_thresh_y = 3

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
            cv2.drawContours(self.calculated_mask, [self.hand_contour], -1, (0, 255, 255), 1)

            if len(hullIndices[0]) > 3:
                self.defects = cv2.convexityDefects(self.hand_contour, hullIndices[0])

            # contourPoints = cv2.convexHull(max_contour, True)
            for index in range(len(contourPoints)):
                cv2.circle(self.calculated_mask, (contourPoints[index][0][0], contourPoints[index][0][1]), 3, (255, 0, 0), -1)
        #

    def find_defects_point(self):



        if type(self.defects) is np.ndarray:
            fingers = 0

            # Get defect points and draw them in the original image
            if self.defects is not None:
                # print('defects shape = ', defects.shape[0])
                for i in range(self.defects.shape[0]):
                    s, e, f, d = self.defects[i, 0]
                    end = start = far = None
                    if len(self.hand_contour) > s:
                        start = tuple(self.hand_contour[s][0])
                    if len(self.hand_contour) > e:
                        end = tuple(self.hand_contour[e][0])
                    if len(self.hand_contour) > f:
                        far = tuple(self.hand_contour[f][0])

                    # cv2.circle(self.calculated_mask, far, 8, [211, 84, 0], -1)
                    #  finger count
                    if end is not None and far is not None and start is not None:
                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                        area = cv2.contourArea(self.hand_contour)

                        if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                            fingers += 1

                            cv2.circle(self.calculated_mask, far, 5, [0, 0, 255], -1)

                if self.mouse_move_index == fingers:
                    self.set_movement_delta()

                if fingers < len(self.move_stats):
                    # if (self.move_stats[fingers] == 0 or self.move_stats[fingers] == self.move_cap):
                    if self.move_stats[fingers] == 0:
                        self.move_stats = [0, 0, 0, 0, 0]
                        self.move_stats[fingers] = 1

                    else:
                        # if self.move_stats[fingers] != self.move_cap:
                        self.move_stats[fingers] += 1

    def get_center(self):
        if self.hand_contour is None:
            return None
        ((x, y), radius) = cv2.minEnclosingCircle(self.hand_contour)
        M = cv2.moments(self.hand_contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return center

    def set_movement_delta(self):
        if self.last_center is None:
            self.last_center = self.get_center()
        current_center = self.get_center()
        dx = current_center[0] - self.last_center[0]
        dy = current_center[1] - self.last_center[1]
        # print(f"dx {dx}, dy {dy}----------")

        self.movement_delta = [0, 0]
        update_flag = False
        if abs(dx) > self.movement_thresh_x:
            self.movement_delta[0] = dx
            update_flag = True

        if abs(dy) > self.movement_thresh_y:
            self.movement_delta[1] = dy
            update_flag = True

        if self.move_stats[2] == 1 or update_flag:
            self.last_center = current_center

    def draw_circle(self):

        if self.hand_contour is not None:

            # c = max(self.cnts, key=cv2.contourArea)
            c = self.hand_contour
            # find the circumcircle of an object
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # Calculates all of the moments up to the third order of a polygon or rasterized shape.
            M = cv2.moments(c)
            # Image moments help you to calculate some features like center of mass of the object, area of the object etc
            if M["m00"]:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                center = None

            if radius > 3 and None:
                cv2.circle(self.calculated_mask, (int(x), int(y)), int(radius), (0, 100, 255), 2)
                cv2.circle(self.calculated_mask, center, 5, (50, 255, 70), -1)

            # self.pts.appendleft(center)
            return center
        return None

    def draw_line(self):

        for i in range(1, len(self.pts)):
            # for i in xrange(1, len(pts)):
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

            thick = int(np.sqrt(len(self.pts) / float(i + 1)) * 2.5)
            cv2.line(self.img, self.pts[i - 1], self.pts[i], (0, 0, 225), thick)

    def show_windows(self):

        cv2.imshow("calculated_mask", self.calculated_mask)
        cv2.imshow("mask", self.mask)
        k = cv2.waitKey(30) & 0xFF

        if k == 32:
            return
