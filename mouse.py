# control movement of the mouse
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
import time


class Mouse(object):

    def __init__(self, image_size):
        self.monitor = get_monitors()[0]
        self.image_size = [image_size[0] - 50, image_size[1] - 50]
        self.image_height = self.image_size[0]
        self.image_width = self.image_size[1]
        self.x_relation = self.monitor.width / self.image_width
        self.y_relation = self.monitor.height / self.image_height
        self.mouse = Controller()
        self.is_left_pressed = False

    def update_view_size(self, image_size):
        if self.image_size != [image_size[0] - 50, image_size[1] - 50]:
            self.image_size = [image_size[0] - 50, image_size[1] - 50]

    def move(self, move_delta):
        x = self.mouse.position[0] + move_delta[0] * self.x_relation
        y = self.mouse.position[1] + move_delta[1] * self.y_relation
        self.mouse.position = (x, y)

    def right_click(self):
        pass

    def left_click(self):
        if not self.is_left_pressed:
            self.is_left_pressed = True
            self.mouse.press(Button.left)

    def release_all(self):
        self.is_left_pressed = False
        self.mouse.release(Button.left)

    def search_trigger(self, move_stats, move_cap, move_delta):
        if move_delta is None:
            move_delta = self.mouse.position

        if move_stats[0] >= move_cap:
            self.release_all()
        if move_stats[1] >= move_cap:
            self.move(move_delta)
            # self.move_mouse(center[0], center[1])
        # if move_stats[3] >= move_cap:
        #     self.release_all()
        if move_stats[4] >= move_cap:
            self.left_click()

    def move_mouse(self, x, y):
        def set_mouse_position(x, y):
            self.mouse.position = (int(x), int(y))

        def smooth_move_mouse(from_x, from_y, to_x, to_y, speed=0.1):
            steps = 40
            sleep_per_step = speed // steps
            x_delta = (to_x - from_x) / steps
            y_delta = (to_y - from_y) / steps
            for step in range(steps):
                new_x = x_delta * (step + 1) + from_x
                new_y = y_delta * (step + 1) + from_y
                set_mouse_position(new_x, new_y)
                time.sleep(sleep_per_step)

        return smooth_move_mouse(
            self.mouse.position[0],
            self.mouse.position[1],
            x * self.x_relation,
            y * self.y_relation
        )

# m = Mouse((100, 100))
# m.move((5, 5))
