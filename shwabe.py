import cv2
from device import Camera
from mouse import Mouse
from image_processor import ImageProcessor
from main_brain import MainBrain
from gui_view import Gui_View


class Shwabe(object):

    def __init__(self):
        self.camera = Camera()
        self.processor = ImageProcessor()
        # self.mouse = Mouse(self.camera.shape[:2])
        self.mouse = Mouse((480,640))
        self.brain = MainBrain()
        self.main_loop_flag = True
        self.idx = 1
        funcs = dict({
            'main_loop_start': self.main_loop,
            'main_loop_stop': self.stop_main_loop,
        })
        self.v = Gui_View(funcs)
        # self.mouse = Mouse(self.camera.get_camera_view()[:2])

    def run(self):
        while True:
            if self.idx % 500 == 0:
                self.v.master.update()
            # self.v.master.mainloop()
            if self.main_loop_flag:
                self.main_loop()

    def main_loop(self):
        self.idx += 1

        while self.main_loop_flag:
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
            self.brain.find_defects_point()

            # self.brain.move_stats
            self.mouse.search_trigger(self.brain.move_stats, self.brain.move_cap, self.brain.movement_delta)
            self.brain.show_windows()

    def stop_main_loop(self):
        self.main_loop_flag = False



shwabe = Shwabe()
shwabe.run()
