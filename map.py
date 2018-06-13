###################################
# Environment of Self Driving Car #
###################################

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

from ai import DQN
Config.set('input', 'mouse', 'mouse, multitouch_on_demand')     # Adding this line if we don't want the right click to put a red point

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0      # last coordinates of x in the last drawing
last_y = 0      # last coordinates of y in the last drawing
n_points = 0    # the total number of points in the last drawing
length = 0      # the length of the last drawing

# Getting our AI, which we call "dqn", and that contains our neural network that represents our Q-function
dqn = DQN(5, 3, 0.9)                    # 5 signals, 3 actions, gamma = 0.9
action2rotation = [0, 15, -15]          # action = 0: no rotation, action = 1, rotate 20 degrees, action = 2, rotate -20 degrees
last_reward = 0                         # initializing the last reward
scores = []                             # initializing the mean score curve (sliding window of the rewards) w.r.t time

# Initializing the map
first_update = True                     # using this trick to initialize the map only once
def init():
    global sand                         # sand is an array that has as many cells as our graphic interface has pixels. Each cell has a one if there is sand, 0 otherwise.
    global goal_x                       # x-coordinate of the goal (where the car has to go, that is the up-left corner or the bot-right corner)
    global goal_y                       # y-coordinate of the goal (where the car has to go, that is the up-left corner or the bot-right corner)
    global first_update
    sand = np.zeros((RIGHT, TOP))       # initializing the sand array with only zeros
    goal_x = 30                         # the goal to reach is at the upper left of the map (the x-coordinate is 20 and not 0 because the car gets bad reward if it touches the wall)
    goal_y = TOP - 30                   # the goal to reach is at the upper left of the map (y-coordinate)
    first_update = False                # trick to initialize the map only once

# Initializing the last distance
last_distance = 0

# Creating the car class
class Car(Widget):

    angle = NumericProperty(0)                                  # the angle between the x-axis and the axis of the direction of the car
    rotation = NumericProperty(0)                               # last rotation, which is either [0, 20, -20] degree(s)
    velocity_x = NumericProperty(0)                             # the vector of coordinates velocity x
    velocity_y = NumericProperty(0)                             # the vector of coordinates velocity y
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)       # detecting if there is any sand in front of the car
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)       # detecting if there is any sand at the left of the car
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)       # detecting if there is any sand at the right of the car
    signal1 = NumericProperty(0)                                # the signal received by sensor1
    signal2 = NumericProperty(0)                                # the signal received by sensor2
    signal3 = NumericProperty(0)                                # the signal received by sensor3

    def move(self, rotation):
        """
        Allowing the car to go straightly, left of right
        self.pos: x = x + v * t
        self.rotation: new rotation, which is either [0, 20, -20] degree(s)
        self.angle: the angle between the x-axis and the axis of the direction of the car
        sensors:
            Vector(30, 0): 30 is the distance between the car and the sensor
        signals:
            For each signal, we take
                (1) all the cells from -10 to +10 of the x coordinates of the sensor and
                (2) all the cells from -10 to +10 of the y coordinates of the sensor.
                Therefore we get the square of 20 x 20 pixels surrounding the sensor.
                Inside the square, we sum all the ones, because the cells contain 0 or 1.
                We divide it by 400 to get the density of ones inside the square and that's how
                we get the signal of the density of centers around the sensor and.
        """
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x) - 10: int(self.sensor1_x) + 10, int(self.sensor1_y) - 10: int(self.sensor1_y) + 10])) / 400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x) - 10: int(self.sensor2_x) + 10, int(self.sensor2_y) - 10: int(self.sensor2_y) + 10])) / 400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x) - 10: int(self.sensor3_x) + 10, int(self.sensor3_y) - 10: int(self.sensor3_y) + 10])) / 400.
        if self.sensor1_x > RIGHT - 10 or self.sensor1_x < 10 or self.sensor1_y > TOP - 10 or self.sensor1_y < 10:
            self.signal1 = 1.                                                                                       # full density of sand, terrible reward
        if self.sensor2_x > RIGHT - 10 or self.sensor2_x < 10 or self.sensor2_y > TOP - 10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > RIGHT - 10 or self.sensor3_x < 10 or self.sensor3_y > TOP - 10 or self.sensor3_y < 10:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class
class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global dqn
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global RIGHT
        global TOP

        RIGHT = self.width
        TOP = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = dqn.update(last_reward, last_signal)
        scores.append(dqn.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y
        last_distance = distance

# Adding the painting tools
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.1171875, 0.53125, 0.65265)
            d = 100.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 30)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            # sand[int(touch.x), int(touch.y)] = 1
            sand[int(touch.x) - 10: int(touch.x) + 10, int(touch.y) - 10: int(touch.y) + 10] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1.
            density = n_points / (length)
            touch.ud['line'].width = int(100 * density + 1)
            sand[int(touch.x) - 20 : int(touch.x) + 20, int(touch.y) - 20 : int(touch.y) + 20] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)
class CarApp(App):
    global parent

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearButton = Button(text = 'CLEAR')
        saveButton = Button(text = 'SAVE', pos = (parent.width, 0))
        loadButton = Button(text = 'LOAD', pos = (2 * parent.width, 0))
        pngButton = Button(text = 'PNG', pos = (3 * parent.width, 0))

        clearButton.bind(on_release = self.clear_canvas)
        saveButton.bind(on_release = self.save)
        loadButton.bind(on_release = self.load)
        pngButton.bind(on_release = self.save_png)

        parent.add_widget(self.painter)
        parent.add_widget(clearButton)
        parent.add_widget(saveButton)
        parent.add_widget(loadButton)
        parent.add_widget(pngButton)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((RIGHT, TOP))

    def save(self, obj):
        print("Saving dqn...")
        dqn.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("Loading last saved dqn...")
        dqn.load()
        
    def save_png(self, obj):
        print("Saving png...")
        self.parent.export_to_png('a.png')

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
    CarApp().run()
