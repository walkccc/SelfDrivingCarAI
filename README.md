# Self Driving Car AI

This is a project of Artificial Intelligence Self Driving Car with 2D simulation.

## Getting Started

Please install both [Kivy](https://kivy.org/#download) and [PyTorch](https://pytorch.org/) to start the project.

## Dependencies

```
Kivy                               1.10.1.dev0
Kivy-Garden                        0.1.4
numpy                              1.14.3
torch                              0.4.0      
torchvision                        0.2.1
Python 3.6.4 :: Anaconda custom (64-bit)
```

## Running the App

```python
python map.py
```

## How to Play

After running the Kivy canvas, you can hold on your left-mouse button to draw the line, therefore the car will be learning the path by [Deep Q-Learning Algorithm](https://en.wikipedia.org/wiki/Q-learning).

## How to combine the files together?

We first import `DQN` class from `ai.py`:

```python
from ai import DQN
```

The we create `dqn` object from the `DQN` class:

```python
dqn = DQN(5, 3, 0.9)    # 5 signals, 3 actions, gamma = 0.9
```

- 5 signals is composed of 

    - three signals of the sensors
    - the orientation
    - the minus orientation

- 3 actions is composed of

    - go straight
    - go left
    - go right

Each time we select the right `action` to play by

```python
action = dqn.update(last_reward, last_signal)
```

where `last_signal` will be the input of the neural network.

```python
last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
```

## Screen Shot

![](./screenshot.png)