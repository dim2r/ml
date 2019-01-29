import pygame, sys
from pygame.locals import *

class CellState():
        OBSTACLE=-1
        MEAL=-2
        ANOTHER_CREATURE=-3
        MY_CREATURE=-4

class Color():
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)


class Board():
    def __init__(self, dimension, resource_percentage):
        self.dimension=dimension
        self.resource_percentage=resource_percentage

        self.cells  =  [[0 for x in range(dimension)] for y in range(dimension)]
        self.creatures =  [[0 for x in range(dimension)] for y in range(dimension)]

