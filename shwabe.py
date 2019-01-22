import cv2
from device import Camera
from mouse import Mouse
from image_processor import ImageProcessor
from main_brain import MainBrain


class Shwabe(object):

    def __init__(self):
        self.camera = Camera()
        self.processor = ImageProcessor()
        # self.mouse = Mouse(self.camera.shape[:2])
        self.mouse = Mouse((480,640))
        self.brain = MainBrain()
        # self.mouse = Mouse(self.camera.get_camera_view()[:2])

    def main_loop(self):
        while True:
            input_frame = self.camera.get_frame()
            # print("===========", input_frame.shape)
            self.mouse.update_view_size(input_frame.shape)
            # self.processor.extract_morph_from_img(input_frame)
            # center = self.processor.draw_circle()
            # if center:
            #     self.mouse.move(center)
            # self.processor.draw_line()
            # self.processor.draw_windows()

            masked_image = self.processor.extract_mask(input_frame)

            self.brain.find_contours(masked_image)
            self.brain.find_defecets_point()

            # self.brain.move_stats
            self.mouse.search_trigger(self.brain.move_stats, self.brain.move_cap, self.brain.get_center())
            # self.brain.show_windows()

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


shwabe = Shwabe()
shwabe.main_loop()
