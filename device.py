import cv2
import numpy as np


class Camera(object):

    def __init__(self):
        self.cam = cv2.VideoCapture(0)

        self.valid = False
        try:
            resp = self.cam.read()
            self.shape = resp[1].shape
            print("===========",self.shape)
            self.valid = True
        except:
            self.shape = None

    def get_frame(self):
        if self.valid:
            ret,frame = self.cam.read()
        else:
            frame = np.ones((480, 640, 3), dtype=np.uint8)
            col = (0, 256, 256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                        (65, 220), cv2.FONT_HERSHEY_PLAIN, 2, col)
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
        cv2.imshow("Original", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# test_camera_class()

