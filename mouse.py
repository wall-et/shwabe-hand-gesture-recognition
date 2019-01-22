# control movement of the mouse
from pynput.mouse import Controller, Button
from screeninfo import get_monitors


class Mouse(object):

    def __init__(self, image_size):
        self.monitor = get_monitors()[0]
        self.image_height = image_size[0]-50
        self.image_width = image_size[1]-50
        self.x_relation = self.monitor.width / self.image_width
        self.y_relation = self.monitor.height / self.image_height
        self.mouse = Controller()
        self.is_left_pressed = False

    def move(self, mouse_location):
        x = mouse_location[0] * self.x_relation
        y = mouse_location[1] * self.y_relation
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

# m = Mouse((100, 100))
# m.move((5, 5))
