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



def MAIN_PYGAME():
    pygame.init()

    windowSurface = pygame.display.set_mode((500, 400), 0, 32)
    pygame.display.set_caption('Hello world!')
    windowSurface.fill(Color.WHITE)
    basicFont = pygame.font.SysFont(None, 55)
    text = basicFont.render('Hello world!!!!', True, Color.WHITE, Color.BLUE)
    windowSurface.blit(text, (0,0))
    pygame.draw.line(windowSurface, Color.BLUE, (60, 60), (120, 60), 4)
    pygame.draw.line(windowSurface, Color.BLUE, (120, 60), (60, 120))
    pygame.draw.line(windowSurface, Color.BLUE, (60, 120), (120, 120), 4)


    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    MAIN_PYGAME()