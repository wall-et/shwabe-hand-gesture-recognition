# control movement of the mouse
from pynput.mouse import Controller
from screeninfo import get_monitors


class Mouse(object):

    def __init__(self, image_size):
        self.monitor = get_monitors()[0]
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.x_relation = self.monitor.width / self.image_width
        self.y_relation = self.monitor.height / self.image_height
        self.mouse = Controller()
        for m in get_monitors():
            print()

    def move(self, mouse_location):
        x = mouse_location[0] * self.x_relation
        y = mouse_location[1] * self.y_relation
        self.mouse.position = (x, y)
        pass

    def right_click(self):
        pass

    def left_click(self):
        pass


# m = Mouse((100, 100))
# m.move((5, 5))
