import cv2
import numpy as np


class Camera:

    def __init__(self):
        self.cam = cv2.VideoCapture(0)

        self.valid = False
        try:
            resp = self.cam.read()
            self.shape = resp[1].shape
            print("===========", self.shape)
            self.valid = True
        except:
            self.shape = None

    def get_camera_view(self):
        return self.get_frame().shape

    def get_frame(self):
        if self.valid:
            ret, frame = self.cam.read()
        else:
            frame = np.ones((480, 640, 3), dtype=np.uint8)
            col = (0, 256, 256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                        (65, 220), cv2.FONT_HERSHEY_PLAIN, 2, col)

        # frame = self.trim_black_edges(frame)
        return frame

    def trim_black_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)

            crop = frame[y:y + h, x:x + w]

            return crop
        return frame

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()


def test_camera_class():
    # cameras = []

    # if camera.valid or not len(cameras):
    #     cameras.append(camera)
    camera = Camera()
    while True:
        frame = camera.get_frame()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)

            crop = frame[y:y + h, x:x + w]
            cv2.imshow("crop", crop)
            # cv2.imwrite('tmp.png', crop)

        cv2.imshow("Original", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# test_camera_class()
