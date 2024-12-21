import pyglet
from gym.envs.classic_control import rendering
import time

class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()

screen_width = 500
screen_height = 500

viewer = rendering.Viewer(screen_width, screen_height + 20)
text = 'hello world'
label = pyglet.text.Label(text, font_size=36,
                          x=100, y=250, anchor_x='left', anchor_y='bottom',
                          color=(255, 123, 255, 255))
label.draw()
viewer.add_geom(DrawText(label))
viewer.render(return_rgb_array=False)
time.sleep(10)