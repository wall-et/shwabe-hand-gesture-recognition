import cv2
from device import Camera
from mouse import Mouse
from image_processor import ImageProcessor


class Shwabe(object):

    def __init__(self):
        self.camera = Camera()
        self.processor = ImageProcessor()
        self.mouse = Mouse(self.camera.shape[:2])

    def main_loop(self):
        while True:
            input_frame = self.camera.get_frame()
            self.processor.extract_morph_from_img(input_frame)
            center = self.processor.draw_circle()
            # if center:
                # self.mouse.move(center)
            self.processor.draw_line()
            self.processor.draw_windows()


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


shwabe = Shwabe()
shwabe.main_loop()