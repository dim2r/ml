import pygame, sys
from pygame.locals import *
import numpy as np


class CellState():
    EMPTY = 0
    OBSTACLE = -1
    MEAL = -2
    ANOTHER_CREATURE = -3
    MY_CREATURE = -4


class CreatureAction():
    NONE = 0
    MOVE_UP=1
    MOVE_DOWN=2
    MOVE_LEFT=3
    MOVE_RIGHT=4

    BITE_SOMETHING_AROUND=5
    BITE_UP=6
    BITE_DOWN=7
    BITE_LEFT=8
    BITE_RIGHT=9


    UP = (0, -1)
    DOWN = (0, +1)
    LEFT = (-1, 0)
    RIGHT = (+1, 0)
    ALL_MOVES = (UP, DOWN, LEFT, RIGHT)
    @staticmethod
    def random_direction():
        i = np.random.randint(0, 4)
        return CreatureAction.ALL_MOVES[i]


class Color():
    BLACK = (0, 0, 0)
    GRAY = (150, 150, 150)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    CREATURE = (200, 100, 100)


class DrawOptions():
    border = 30
    cell_size = 20


class Creature():
    def __init__(self, board):
        self.board = board
        self.board.creature_count += 1
        self.id = board.creature_count
        self.cells = []

        x, y = self.board.find_empty_cell()
        self.board.cells[x][y] = self.id
        self.cells.append([x , y])

    def get_head(self):
        return self.cells[0]
    def get_tail(self):
        return self.cells[len(self.cells)-1]

    def can_move(self,direction):
        x,y = self.get_head()
        x += direction[0]
        y += direction[1]

        if x>=0 and x<self.board.dimension and y>=0 and y<self.board.dimension and self.board.cells[x][y]==CellState.EMPTY:
            return True
        else:
            return False

    def move(self, direction):
        if self.can_move(direction):
            x,y = self.get_head()
            x += direction[0]
            y += direction[1]

            self.cells.insert(0,[x,y])
            self.board.cells[x][y]=self.id

            xx,yy = self.get_tail()
            self.board.cells[xx][yy]=CellState.EMPTY
            del self.cells[-1]

    def random_move(self):
        dir = CreatureAction.random_direction()
        self.move(dir)

    def random_bite(self):
        x,y = self.get_head()

        direction = CreatureAction.random_direction()
        x += direction[0]
        y += direction[1]

        if self.board.is_inside(x,y) and self.board.cells[x][y]==CellState.MEAL:
            self.cells.insert(0,[x,y])
            self.board.cells[x][y]=self.id



class Board():
    def __init__(self, dimension, resource_percentage):
        self.dimension = dimension
        self.resource_percentage = resource_percentage

        self.cells = [[0 for x in range(dimension)] for y in range(dimension)]
        self.creatures = []
        self.creature_count = 0
        for i in range(self.dimension):
            self.cells[i][0]=CellState.OBSTACLE
            self.cells[0][i]=CellState.OBSTACLE
            self.cells[i][dimension-1]=CellState.OBSTACLE
            self.cells[dimension-1][i]=CellState.OBSTACLE

    def is_inside(self,x,y):
        return x>=0 and x<self.dimension and y>=0 and y<self.dimension

    def add_creatures(self, count):
        for i in range(count):
            c = Creature(self)
            self.creatures.append(c)


    def add_meal(self, count):
        for i in range(count):
            x, y = self.find_empty_cell()
            self.cells[x][y] = CellState.MEAL

    def find_empty_cell(self):
        attempt = 0

        while True:
            x = np.random.randint(0, self.dimension)
            y = np.random.randint(0, self.dimension)
            attempt += 1

            if attempt > 100:
                raise BaseException("cannot place a creature")

            if self.cells[x][y] == CellState.EMPTY:
                return x, y

    # end def find_empty_cell()

    def step(self):
        for c in self.creatures:
            c.random_move()
            c.random_bite ()

    def draw(self, windowSurface):
        for i in range(self.dimension + 1):
            pygame.draw.line(windowSurface, Color.GRAY,
                             (DrawOptions.border + i * DrawOptions.cell_size
                              , DrawOptions.border
                              ), (DrawOptions.border + i * DrawOptions.cell_size
                                  , DrawOptions.border + self.dimension * DrawOptions.cell_size
                                  ), 1)  # end line

            pygame.draw.line(windowSurface, Color.GRAY,
                             (DrawOptions.border
                              , DrawOptions.border + i * DrawOptions.cell_size
                              ), (DrawOptions.border + self.dimension * DrawOptions.cell_size
                                  , DrawOptions.border + i * DrawOptions.cell_size
                                  ), 1)  # end line

        for x in range(self.dimension):
            for y in range(self.dimension):

                color = None
                if self.cells[x][y] > 0:
                    color = Color.CREATURE

                if self.cells[x][y] == CellState.OBSTACLE:
                    color = Color.BLACK
                if self.cells[x][y] == CellState.MEAL:
                    color = Color.GREEN

                if color is not None:
                    pygame.draw.rect(windowSurface
                                     , color
                                     , (DrawOptions.border + x * DrawOptions.cell_size + 2
                                        , DrawOptions.border + y * DrawOptions.cell_size + 2
                                        , DrawOptions.cell_size - 3
                                        , DrawOptions.cell_size - 3
                                        )
                                     )  # end rect

        for c in self.creatures:
            x,y = c.get_head()
            pygame.draw.rect(windowSurface
                             ,   Color.WHITE
                             , (DrawOptions.border + x * DrawOptions.cell_size + 4
                                , DrawOptions.border + y * DrawOptions.cell_size + 4
                                , DrawOptions.cell_size - 7
                                , DrawOptions.cell_size - 7
                                )
                             )  # end rect
        # end for i,j
        # end draw


def MAIN_PYGAME():
    board = Board(50, 0.1)
    draw_len = DrawOptions.cell_size * board.dimension + DrawOptions.border * 2
    board.add_creatures(11)
    board.add_meal(333)

    pygame.init()

    windowSurface = pygame.display.set_mode((draw_len, draw_len), 0, 32)
    pygame.display.set_caption('Snakes with Reinforcement')
    windowSurface.fill(Color.WHITE)
    basicFont = pygame.font.SysFont(None, 30)
    text = basicFont.render('  Snake world  ', True, Color.WHITE, Color.BLUE)
    windowSurface.blit(text, (0, 0))

    board.draw(windowSurface)
    pygame.display.update()

    MOVEEVENT, t = pygame.USEREVENT+1, 2
    pygame.time.set_timer(MOVEEVENT, t)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN or  event.type == MOVEEVENT:
                board.step()
                windowSurface.fill(Color.WHITE)
                board.draw(windowSurface)
                pygame.display.update()


if __name__ == "__main__":
    MAIN_PYGAME()
