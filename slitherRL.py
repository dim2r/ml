import  pygame, sys
from    pygame.locals import *
import  numpy as np


class CellState():
    EMPTY = 0
    OBSTACLE = -1
    MEAL = -2
    ANOTHER_CREATURE = -3
    ANOTHER_CREATURE_HEAD = -4
    MY_CREATURE = -5
    DEAD = -6
    OUTSIDE = -100


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

    ALL_DIRECTIONS = ( (-1,-1),(0,-1),(1,-1), (-1,0),(1,0),  (-1,1),(0,1),(1,1) ) #(0,0) excluded

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
    CREATURE = (200, 70, 70)
    DEAD = (150, 80, 80)



class DrawOptions():
    border = 30
    cell_size = 13


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

        if self.board.is_inside(x,y) and (self.board.cells[x][y]==CellState.MEAL or self.board.cells[x][y]==CellState.DEAD):
            self.cells.insert(0,[x,y])
            self.board.cells[x][y]=self.id
            return

        if self.board.is_inside(x,y) and self.board.cells[x][y]>0:
            id=self.board.cells[x][y]
            if self.id!=id or np.random.randint(0, 100)<3:
                for c in self.board.creatures:

                    if c.id == id and len(c.cells)>3:
                        cut_i = -1

                        for i in range(2,len(c.cells)):
                            cell_x  =c.cells[i][0]
                            cell_y = c.cells[i][1]
                            if cell_x==x and cell_y==y:
                                cut_i=i


                            if cut_i>0:
                                self.board.cells[cell_x][cell_y] = CellState.DEAD
                        #end for
                        if cut_i>=0:
                            c.cells = c.cells[:(cut_i)]



    def get_sensor_data(self):

        head = self.get_head()

        for dx in range(-2,2):
            for dy in range(-2,2):
                if not(dx==0 and dy==0):
                    x=head[0]+dx
                    y=head[0]+dy



        # pass
        # radii_start=1
        # super_cell_size = 1
        # for radii_level in range(1,3):
        #     for dir in CreatureAction.ALL_DIRECTIONS:
        #         for dx in range(radii_start,radii_start+super_cell_size-1)
        #             for dy in range(radii_start,radii_start+super_cell_size-1)
        # super_cell_size  = super_cell_size * 2


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

    def get_cell_type(self,x,y, my_id):
        if not self.is_inside(x,y):
            return CellState.OBSTACLE

        else:
            c = self.cells[x][y]
            if c>0 and c==my_id:
                return CellState.MY_CREATURE
            if c>0 and c!=my_id:
                return CellState.ANOTHER_CREATURE
            if c==CellState.DEAD or c==CellState.DEAD:
                return CellState.MEAL
        return CellState.OBSTACLE



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
                if self.cells[x][y] == CellState.DEAD:
                    color = Color.DEAD

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
    board.add_creatures(5)
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

    MOVEEVENT, t = pygame.USEREVENT+1, 20
    pygame.time.set_timer(MOVEEVENT, t)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN or  event.type == MOVEEVENT:
                pygame.event.clear()
                board.step()
                windowSurface.fill(Color.WHITE)
                board.draw(windowSurface)
                pygame.display.update()
                pygame.event.clear()


if __name__ == "__main__":
    MAIN_PYGAME()
