import torch
import torch.nn as nn
import numpy as np
from enum import IntEnum
# import sys
# import os.path
import os
import socket

host_name = socket.gethostname()

in_dimension = 3
in_win_row_len = 3
in_batch_count = 200

in_flat_dimension = in_dimension * in_dimension
in_save_file_name = 'tictactoe_' + str(host_name) + '_'
in_save_file_name += str(in_dimension) + str(in_win_row_len)
in_learning_rate = 0.00001  # successfull batch=200 lr=0.0003 for relu-sigmoid
in_learning_rate_discount = 0.99

in_episode_move_discount = 1

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


class NeuralNet(nn.Module):
    def __init__(self, dimension, marker_char):
        super(NeuralNet, self).__init__()
        flatdimension = dimension * dimension
        self.net = nn.Sequential(
            nn.Linear(flatdimension, flatdimension)
            , nn.Sigmoid()
            # , nn.Linear(flatdimension, flatdimension)
            # , nn.Sigmoid()
            # , nn.Linear(flatdimension, flatdimension)
            # , nn.LeakyReLU()
            , nn.Linear(flatdimension, flatdimension)
            , nn.Sigmoid()
        ).to(device)
        self.marker_char = marker_char
        self.abbr = 'SS'

    def forward(self, x):
        return self.net(x)

    def get_file_name(self):
        return in_save_file_name + '_' + self.abbr + '_(' + self.marker_char + ')'

    def save(self, count=None):
        try:
            torch.save(self.state_dict(), self.get_file_name())
        except:
            pass
        if not count is None:
            try:
                torch.save(self.state_dict(), self.get_file_name() + '-' + str(count) + '-')
            except:
                pass

    def load(self):
        self.load_state_dict(torch.load(self.get_file_name()))


class CellState(IntEnum):
    EMPTY = 0
    X = 1
    O = -1


class BoardState(IntEnum):
    X_win = CellState.X
    O_win = CellState.O
    DRAW = 0
    INPROGRESS = 2


class Board():
    def __init__(self, dimension, win_row_len):
        self.dimension = dimension
        self.data = np.zeros(dimension * dimension, dtype=int)
        self.marked_count = 0
        self.win_row_len = win_row_len
        self.debug = False
        self.last_x = -1
        self.last_y = -1

    def reset(self):
        self.data = np.zeros(self.dimension * self.dimension, dtype=int)
        self.marked_count = 0

    def flat(self, x, y):
        return x + y * self.dimension

    def unflat2d(self, flat_index):
        y = flat_index // self.dimension
        x = flat_index - y * self.dimension
        return x, y

    def rotate90(self, x, y):
        return self.dimension - 1 - y, x

    def rotate180(self, x, y):
        return self.dimension - 1 - x, self.dimension - 1 - y

    def rotate270(self, x, y):
        return y, self.dimension - 1 - x

    def flip1(self, x, y):
        return self.dimension - 1 - x, y

    def flip2(self, x, y):
        return x, self.dimension - 1 - y

    def flip3(self, x, y):
        return self.dimension - 1 - x, self.dimension - 1 - y

    def transform_data(self, transform):  # rotate90,rotate180,rotate270,flip1,flip2,flip3
        res = np.zeros_like(self.data)
        for x in range(self.dimension):
            for y in range(self.dimension):
                xx, yy = transform(x, y)
                res[self.flat(xx, yy)] = self.data[self.flat(x, y)]
        return res

    def isinside(self, x, y):
        return (x >= 0 and y >= 0 and x < self.dimension and y < self.dimension)

    def put_marker(self, x, y, marker):
        if self.debug:
            print('put {} at {},{}'.format(marker, x, y))
        f = self.flat(x, y)
        if self.data[f] == CellState.EMPTY:
            self.data[f] = marker
            self.marked_count += 1
            self.last_x = x
            self.last_y = y
        else:
            self.print()
            raise BaseException("cell {},{} is not empty".format(x, y))

    def get(self, x, y):
        return self.data[self.flat(x, y)]

    def markerToChar(self, marker):
        if marker == CellState.EMPTY:
            return "-"
        if marker == CellState.X:
            return "X"
        if marker == CellState.O:
            return "O"
        return '?'

    def getCharAt(self, x, y):
        return self.markerToChar(self.get(x, y))

    def print(self, file=None):

        out = ""
        out += '  __________'[0:self.dimension + 2]
        out += '\n'
        for y in range(self.dimension):
            s = str(y + 1) + "|"
            for x in range(self.dimension):
                if (x == self.last_x) and (y == self.last_y):
                    s += self.getCharAt(x, y).upper()
                else:
                    s += self.getCharAt(x, y).lower()
            out += (s + "|")
            out += '\n'
        out += ('  ABCDEFGHIJ'[0:self.dimension + 2])
        out += '\n'

        state, x, y = self.board_state()
        # print('marked_count={}   '.format( self.marked_count))
        if state != BoardState.INPROGRESS:
            if state == BoardState.DRAW:
                out += ("DRAW")
                out += '\n'
            else:
                out += ("winner='{}' x={} y={}".format(self.markerToChar(state), x, y))
                out += '\n'
        if file is None:
            print(out)
        else:
            with open(file, 'a') as f:
                print(out, file=f)

    def board_state(self):
        for x in range(self.dimension):
            for y in range(self.dimension):
                if self.get(x, y) != CellState.EMPTY:
                    current_marker = self.get(x, y)
                    count1 = 1
                    count2 = 1
                    count3 = 1
                    count4 = 1
                    for delta in range(1, self.win_row_len):
                        xx = x + delta
                        yy = y + delta
                        if self.isinside(xx, yy) and self.get(xx, yy) == current_marker:
                            count1 += 1

                        xx = x - delta
                        yy = y + delta
                        if self.isinside(xx, yy) and self.get(xx, yy) == current_marker:
                            count2 += 1

                        xx = x + 0
                        yy = y + delta
                        if self.isinside(xx, yy) and self.get(xx, yy) == current_marker:
                            count3 += 1
                        xx = x + delta
                        yy = y + 0
                        if self.isinside(xx, yy) and self.get(xx, yy) == current_marker:
                            count4 += 1

                    if count1 >= in_win_row_len:
                        return current_marker, x, y
                    if count2 >= in_win_row_len:
                        return current_marker, x, y
                    if count3 >= in_win_row_len:
                        return current_marker, x, y
                    if count4 >= in_win_row_len:
                        return current_marker, x, y

        if self.marked_count == self.dimension * self.dimension:
            return BoardState.DRAW, -1, -1

        return BoardState.INPROGRESS, -1, -1

    def another_marker(self, marker):
        return -marker

    def can_make_turn(self):
        return (self.marked_count < self.dimension * self.dimension)

    def random_move(self, marker):
        if not self.can_make_turn():
            self.print()
            print("cannot make a turn")
            raise BaseException("cannot make a turn")
        attempt = 0
        while True:
            x = np.random.randint(0, self.dimension)
            y = np.random.randint(0, self.dimension)
            if self.get(x, y) == CellState.EMPTY:
                self.put_marker(x, y, marker)
                return x, y
            attempt += 1
            if attempt > 1000:
                self.print()
                raise BaseException('infinite loop')


class PlayerAgent():
    def __init__(self, board, marker):
        self.board = board
        self.marker = marker
        self.exit = False
        self.win_count = 0
        self.draw_count = 0
        self.max_win_count = 0
        self.random_move_count = 0
        self.move_count = 0
        self.debug = False

    def make_move(self):
        pass

    def init_statistics(self):
        self.win_count = 0
        self.draw_count = 0
        self.random_move_count = 0
        self.move_count = 0


class RandomPlayerAgent(PlayerAgent):
    def __init__(self, board, marker):
        super(RandomPlayerAgent, self).__init__(board, marker)

    def make_move(self):
        self.board.random_move(self.marker)


class BotPlayerAgent(PlayerAgent):
    def __init__(self, board, marker):
        super(BotPlayerAgent, self).__init__(board, marker)
        self.random_move_percent = 0.20

    def dbg(self, msg):
        if self.debug:
            print(msg)

    def make_move(self):
        if np.random.random_sample() > 1 - self.random_move_percent:
            self.dbg('random move1')
            self.board.random_move(self.marker)
            return

        collect_my_win_moves = []
        collect_my_pre_win_moves = []

        for x in range(self.board.dimension):
            for y in range(self.board.dimension):
                if self.board.get(x, y) == CellState.EMPTY:
                    board2 = Board(self.board.dimension, self.board.win_row_len)
                    board2.data = self.board.data.copy()
                    board2.put_marker(x, y, self.marker)
                    state, xx, yy = board2.board_state()
                    if state == BoardState.X_win or state == BoardState.O_win:
                        # board.put_marker(x, y, self.marker)
                        collect_my_win_moves.append([x, y])
                    if self.board.dimension > 3 and self.board.marked_count > 2:
                        for x2 in range(self.board.dimension):
                            for y2 in range(self.board.dimension):
                                if board2.get(x2, y2) == CellState.EMPTY:
                                    board3 = Board(self.board.dimension, self.board.win_row_len)
                                    board3.data = board2.data.copy()
                                    board3.put_marker(x2, y2, self.marker)
                                    state, xx, yy = board3.board_state()
                                    if state == BoardState.X_win or state == BoardState.O_win:
                                        collect_my_pre_win_moves.append([x, y])

        if len(collect_my_win_moves) > 0:
            self.dbg('my_win_moves ' + str(collect_my_win_moves))
            i = np.random.randint(0, len(collect_my_win_moves))
            x = collect_my_win_moves[i][0]
            y = collect_my_win_moves[i][1]
            self.board.put_marker(x, y, self.marker)
            return  # priority my win

        collect_his_win_moves = []
        collect_his_pre_win_moves = []
        his_marker = board2.another_marker(self.marker)
        for x in range(self.board.dimension):
            for y in range(self.board.dimension):
                if self.board.get(x, y) == CellState.EMPTY:
                    board2 = Board(self.board.dimension, self.board.win_row_len)
                    board2.data = self.board.data.copy()
                    board2.put_marker(x, y, his_marker)  ###diff<<<
                    state, xx, yy = board2.board_state()
                    if state == BoardState.X_win or state == BoardState.O_win:
                        collect_his_win_moves.append([x, y])
                    if self.board.dimension > 3 and self.board.marked_count > 2:
                        for x2 in range(self.board.dimension):
                            for y2 in range(self.board.dimension):
                                if board2.get(x2, y2) == CellState.EMPTY:
                                    board3 = Board(self.board.dimension, self.board.win_row_len)
                                    board3.data = board2.data.copy()
                                    board3.put_marker(x2, y2, his_marker)  ###diff<<<
                                    state, xx, yy = board3.board_state()
                                    if state == BoardState.X_win or state == BoardState.O_win:
                                        collect_his_pre_win_moves.append([x, y])

        if len(collect_his_win_moves) > 0:
            self.dbg('collect_his_win_moves ' + str(collect_his_win_moves))
            i = np.random.randint(0, len(collect_his_win_moves))
            x = collect_his_win_moves[i][0]
            y = collect_his_win_moves[i][1]
            self.board.put_marker(x, y, self.marker)
            return

        if len(collect_my_pre_win_moves) > 0:
            self.dbg('collect_my_pre_win_moves ' + str(collect_my_pre_win_moves))
            i = np.random.randint(0, len(collect_my_pre_win_moves))
            x = collect_my_pre_win_moves[i][0]
            y = collect_my_pre_win_moves[i][1]
            self.board.put_marker(x, y, self.marker)
            return

        if len(collect_his_pre_win_moves) > 0:
            self.dbg('collect_his_pre_win_moves ' + str(collect_his_pre_win_moves))
            i = np.random.randint(0, len(collect_his_pre_win_moves))
            x = collect_his_pre_win_moves[i][0]
            y = collect_his_pre_win_moves[i][1]
            self.board.put_marker(x, y, self.marker)
            return

        mid = self.board.dimension // 2
        if self.board.get(mid, mid) == CellState.EMPTY:
            self.dbg('center mark')
            self.board.put_marker(mid, mid, self.marker)
            return

        # maxx = self.board.dimension - 1
        corners = [
            [mid - 1, mid - 1]
            , [mid - 1, mid + 1]
            , [mid + 1, mid - 1]
            , [mid + 1, mid + 1]
        ]
        free_corner = []
        for xy in corners:
            x = xy[0]
            y = xy[1]
            xx, yy = self.board.rotate180(x, y)
            opposite_cell = self.board.get(xx, yy)
            if self.board.get(x, y) == CellState.EMPTY and (
                    opposite_cell == CellState.EMPTY or opposite_cell == his_marker):
                free_corner.append(xy)

        if len(free_corner) > 0:
            i = np.random.randint(0, len(free_corner))
            x = free_corner[i][0]
            y = free_corner[i][1]
            self.board.put_marker(x, y, self.marker)
            return

        self.dbg('final random move')
        self.board.random_move(self.marker)
        return


class InputPlayerAgent(PlayerAgent):
    def __init__(self, board, marker):
        super(InputPlayerAgent, self).__init__(board, marker)
        self.bot_player_agent = BotPlayerAgent(board, marker)
        self.bot_player_agent.debug = True
        self.bot_player_agent.random_move_percent = 0.00
        self.move_no = 0

    def make_move(self):
        self.move_no += 1
        if self.move_no == 1:
            print('Help: 2B - put marker to 2B cell; a-automatic move; r-random move; q-quit;')

        try_no = 0
        while True:
            try:

                self.exit = False
                try_no += 1
                if try_no > 1:
                    print('try again')
                inp = input(":>")  ###wait
                inp = inp.upper().strip()

                if inp == 'R':
                    self.board.random_move(self.marker)
                    return
                if inp == 'A':
                    self.bot_player_agent.make_move()
                    return
                if inp == 'Q':
                    self.exit = True
                    return

                x1 = ord(inp[0:1]) - ord('A')
                y1 = ord(inp[1:2]) - ord('1')

                x2 = ord(inp[1:2]) - ord('A')
                y2 = ord(inp[0:1]) - ord('1')

                if (x1 in range(10) and y1 in range(10)):
                    x = x1
                    y = y1
                if (x2 in range(10) and y2 in range(10)):
                    x = x2
                    y = y2

                if self.board.get(x, y) == CellState.EMPTY:
                    self.board.put_marker(x, y, self.marker)
                    return
                else:
                    print('try again')
            except:
                pass


class NeuralNetPlayerAgent(PlayerAgent):
    def __init__(self, board, marker):
        super(NeuralNetPlayerAgent, self).__init__(board, marker)
        self.neural_net = NeuralNet(self.board.dimension, self.board.markerToChar(marker))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.neural_net.parameters(), lr=in_learning_rate)
        self.episode_data = []
        self.stochastic_move = False
        self.curiosity = 0

    def init_episode(self):
        self.episode_data = []

    def store_move(self, my_move_label_flat_i):
        self.episode_data.append({'board_data': self.board.data.copy(), 'my_move': my_move_label_flat_i})

    def apply_gradient_by_episode(self, reward_value):
        discount = in_episode_move_discount

        uniq_array_signatures = []

        def get_signature(arr):
            result = ''
            for a in arr:
                result += str(a)
            return result

        def is_uniq(arr):
            signature = get_signature(arr)
            if not (signature in uniq_array_signatures):
                uniq_array_signatures.append(signature)
                return True
            else:
                return False

        def add_transformed(transform_func, board_data_orig, my_move_orig):
            board_data2 = np.zeros_like(board_data_orig)
            for x in range(self.board.dimension):
                for y in range(self.board.dimension):
                    xx, yy = transform_func(x, y)
                    board_data2[self.board.flat(xx, yy)] = board_data_orig[self.board.flat(x, y)]

            x, y = self.board.unflat2d(my_move_orig)
            xx, yy = transform_func(x, y)
            my_move2 = self.board.flat(xx, yy)

            if is_uniq(board_data2):
                data_augmented_batch.append(board_data2)
                my_move_augmented_batch.append(my_move2)

        for i in range(len(self.episode_data)):
            data_augmented_batch = []
            my_move_augmented_batch = []

            board_data_orig = self.episode_data[i]['board_data']
            my_move_orig = self.episode_data[i]['my_move']
            data_augmented_batch.append(board_data_orig)
            my_move_augmented_batch.append(my_move_orig)

            if np.random.random_sample() > 0.5:
                uniq_array_signatures.append(get_signature(board_data_orig))
                add_transformed(self.board.rotate90, board_data_orig, my_move_orig)
                add_transformed(self.board.rotate180, board_data_orig, my_move_orig)
                add_transformed(self.board.rotate270, board_data_orig, my_move_orig)
                add_transformed(self.board.flip1, board_data_orig, my_move_orig)
                add_transformed(self.board.flip2, board_data_orig, my_move_orig)
                add_transformed(self.board.flip3, board_data_orig, my_move_orig)

            input = torch.tensor(data_augmented_batch, dtype=torch.float, device=device)
            output = self.neural_net(input)
            labels = torch.tensor(my_move_augmented_batch, dtype=torch.long, device=device)
            loss = discount * reward_value * self.criterion(output, labels)
            loss.backward()
        self.optimizer.step()

        discount *= discount

    def make_move(self):
        if not self.board.can_make_turn():
            self.board.print()
            print("cannot make a turn")
            raise BaseException("cannot make a turn")
        self.move_count += 1

        if self.curiosity > 0 and np.random.random_sample() < self.curiosity:
            x, y = self.board.random_move(self.marker)
            self.store_move(self.board.flat(x, y))
            return

        input = torch.tensor([self.board.data], dtype=torch.float, device=device)
        output = self.neural_net(input)
        output = output[0].data.cpu().numpy()

        if self.stochastic_move == False:
            max_output = -1e10
            max_i = -1
            for i in range(len(output)):
                if self.board.data[i] == CellState.EMPTY:
                    if max_output < output[i]:
                        max_i = i
                        max_output = output[i]

        if self.stochastic_move == True:
            output_for_free_cells = output.copy()
            for i in range(len(output_for_free_cells)):
                if self.board.data[i] != CellState.EMPTY:
                    output_for_free_cells[i] = 0

            summ = sum(output_for_free_cells)
            rnd = summ * np.random.random_sample()
            s = 0
            max_i = -1
            for i in range(len(output_for_free_cells)):
                s += output_for_free_cells[i]
                if s >= rnd:
                    max_i = i
                    break

        if max_i == -1:
            raise BaseException('can not choose a move')

        if self.board.data[max_i] == CellState.EMPTY:
            x, y = self.board.unflat2d(max_i)
            self.board.put_marker(x, y, self.marker)
            self.store_move(max_i)
        else:
            x, y = self.board.random_move(self.marker)
            max_i = self.board.flat(x, y)
            self.store_move(max_i)
            self.random_move_count += 1


def play(board, agent1, agent2, do_print=True):
    board.reset()
    current_marker = CellState.X
    agent1.exit = False
    agent2.exit = False
    while board.can_make_turn():
        if do_print:
            board.print()

        if agent1.marker == current_marker:
            agent = agent1
        else:
            agent = agent2

        if do_print:
            print(agent.__class__.__name__ + " is putting " + board.markerToChar(current_marker) + ':')

        agent.make_move()#

        if agent.exit:
            return

        state, x, y = board.board_state()#

        if state == BoardState.X_win or state == BoardState.O_win:#
            if agent1.marker == state:
                agent1.win_count += 1
            if agent2.marker == state:
                agent2.win_count += 1

            if do_print:
                board.print()
                print(agent.__class__.__name__)
            return state#
            break

        if state == BoardState.DRAW:
            agent1.draw_count += 1
            agent2.draw_count += 1
            if do_print:
                board.print()
            return state#
            break

        current_marker = board.another_marker(current_marker)


# test
if False:
    board = Board(in_dimension, in_win_row_len)
    board.put_marker(0, 0, 1)
    board.put_marker(0, 1, -1)
    board.print()
    board.data = board.rotate_data(board.flip3)
    board.print()
    sys.exit()

# test1
if False:
    board = Board(in_dimension, in_win_row_len)
    board.put_marker(0, 0, 1)
    board.put_marker(0, 1, 1)
    board.put_marker(1, 2, 1)
    board.put_marker(2, 0, -1)
    board.put_marker(2, 1, 1)
    board.put_marker(1, 0, -1)
    board.put_marker(1, 1, -1)
    board.put_marker(0, 2, -1)
    board.put_marker(2, 2, -1)
    board.print()
    state, x, y = board.board_state()
    print(state)
    sys.exit()


def train_and_save_best_NNvsBot():
    board = Board(in_dimension, in_win_row_len)

    neural_net_player_agent = NeuralNetPlayerAgent(board, CellState.X)
    try:
        neural_net_player_agent.neural_net.load()
        print('loaded "' + in_save_file_name + '"')
    except:
        pass

    bot_player_agent = BotPlayerAgent(board, CellState.O)

    learning_rate = in_learning_rate
    max_OK_game_count = 0
    episode_no = 0

    while True:
        print('learning ' + str(in_dimension) + 'x' + str(in_dimension) + ' row len=' + str(in_win_row_len))
        X_win_count = 0
        O_win_count = 0
        DRAW_count = 0
        load_count = 0
        for epoch in range(100):

            learning_rate *= in_learning_rate_discount
            neural_net_player_agent.optimizer = torch.optim.Adam(neural_net_player_agent.neural_net.parameters(),
                                                                 lr=learning_rate)

            for i in range(in_batch_count):
                episode_no += 1

                neural_net_player_agent.init_episode()
                state = None

                if np.random.random_sample() < 0.5:
                    bot_player_agent.random_move_percent = 0.50
                else:
                    bot_player_agent.random_move_percent = 0.00

                board.reset()
                current_marker = CellState.X
                while board.can_make_turn():

                    if board.can_make_turn():
                        if neural_net_player_agent.marker == current_marker:
                            neural_net_player_agent.make_move()
                        else:
                            bot_player_agent.make_move()
                            # board.random_move(marker)

                    state, x, y = board.board_state()

                    if state == BoardState.X_win:
                        X_win_count += 1
                        neural_net_player_agent.apply_gradient_by_episode(+1)
                        break
                    if state == BoardState.O_win:
                        O_win_count += 1
                        neural_net_player_agent.apply_gradient_by_episode(-1)
                        break
                    if state == BoardState.DRAW:
                        DRAW_count += 1
                        neural_net_player_agent.apply_gradient_by_episode(+0.5)
                        break

                    current_marker = board.another_marker(current_marker)

            sss = ''
            if (X_win_count + DRAW_count) > max_OK_game_count:
                max_OK_game_count = (X_win_count + DRAW_count)
                sss = 'save'
                load_count = 0
                neural_net_player_agent.neural_net.save(max_OK_game_count)


            else:

                sss = ''
                load_count += 1
                if load_count > 100:
                    break

                if load_count <= 50:
                    try:
                        neural_net_player_agent.neural_net.load()
                        sss = ' ' + str(load_count)
                    except:
                        neural_net_player_agent = NeuralNetPlayerAgent(board, CellState.X)
                        sss = 'try new net '

                learning_rate = in_learning_rate * np.random.random_sample() * 10
                if np.random.random_sample() > 0.9:
                    learning_rate = in_learning_rate * 100
                if np.random.random_sample() > 0.99:
                    learning_rate = in_learning_rate * 1000

            print(
                '{:>6} X_win_count={:<3} DRAW_count={:<3} O_win_count={:<3} (learning_rate={:.10f}) (max win+draw={}) {}'.format(
                    episode_no,
                    X_win_count,
                    DRAW_count,
                    O_win_count,
                    learning_rate,
                    max_OK_game_count,
                    sss))

            X_win_count = 0
            O_win_count = 0
            DRAW_count = 0


def train_and_save_best_NNvsNN():
    board = Board(in_dimension, in_win_row_len)

    neural_net_player_agent1 = NeuralNetPlayerAgent(board, CellState.X)
    neural_net_player_agent1.stochastic_move = True

    try:
        neural_net_player_agent1.neural_net.load()
        print('loaded "' + neural_net_player_agent1.neural_net.get_file_name() + '"')
    except:
        pass

    neural_net_player_agent2 = NeuralNetPlayerAgent(board, CellState.O)
    neural_net_player_agent2.stochastic_move = True
    try:
        neural_net_player_agent2.neural_net.load()
        print('loaded "' + neural_net_player_agent2.neural_net.get_file_name() + '"')
    except:
        pass

    learning_rate = in_learning_rate
    max_OK_game_count = 0
    episode_no = 0

    while True:
        print('learning ' + str(in_dimension) + 'x' + str(in_dimension) + ' row len=' + str(in_win_row_len))
        for epoch in range(100):

            learning_rate *= in_learning_rate_discount

            neural_net_player_agent1.init_statistics()
            neural_net_player_agent2.init_statistics()
            neural_net_player_agent1.neural_net.save()
            neural_net_player_agent2.neural_net.save()
            for i in range(in_batch_count):
                play(board, neural_net_player_agent1, neural_net_player_agent2)

            win_count1 = neural_net_player_agent1.win_count
            win_count2 = neural_net_player_agent2.win_count
            draw_count = neural_net_player_agent1.draw_count

            if epoch == 0:
                who_is_learning = neural_net_player_agent1
            elif neural_net_player_agent1.win_count < neural_net_player_agent2.win_count and np.random.random_sample() > 0.2:
                who_is_learning = neural_net_player_agent1
            elif neural_net_player_agent1.win_count > neural_net_player_agent2.win_count and np.random.random_sample() > 0.2:
                who_is_learning = neural_net_player_agent2
            elif np.random.random_sample() > 0.5:
                who_is_learning = neural_net_player_agent1
            else:
                who_is_learning = neural_net_player_agent2

            neural_net_player_agent1.init_statistics()
            neural_net_player_agent2.init_statistics()

            learning_rate = in_learning_rate
            # learning_rate = in_learning_rate * np.random.random_sample() * 10
            # if np.random.random_sample() > 0.9:
            #     learning_rate = in_learning_rate * 100
            # if np.random.random_sample() > 0.99:
            #     learning_rate = in_learning_rate * 1000

            who_is_learning.optimizer = torch.optim.Adam(who_is_learning.neural_net.parameters(),
                                                         lr=learning_rate)

            neural_net_player_agent1.curiosity = 0.00
            neural_net_player_agent2.curiosity = 0.00
            if np.random.random_sample() > 0.5:
                who_is_learning.curiosity = 0.50
            else:
                who_is_learning.curiosity = 0.00

            for i in range(in_batch_count):
                episode_no += 1

                board.reset()
                neural_net_player_agent1.init_episode()
                neural_net_player_agent2.init_episode()
                current_marker = CellState.X

                while board.can_make_turn():

                    if board.can_make_turn():
                        if neural_net_player_agent1.marker == current_marker:
                            neural_net_player_agent1.make_move()
                        else:
                            neural_net_player_agent2.make_move()

                    if i == in_batch_count - 1:
                        try:
                            board.print('last_game.txt')
                        except:
                            pass

                    if board.marked_count >= in_win_row_len:
                        state, x, y = board.board_state()

                        if state == BoardState.X_win:
                            neural_net_player_agent1.win_count += 1

                            if who_is_learning.marker == CellState.X:
                                who_is_learning.apply_gradient_by_episode(+1)
                            else:
                                who_is_learning.apply_gradient_by_episode(-1)

                            break

                        if state == BoardState.O_win:
                            neural_net_player_agent2.win_count += 1

                            if who_is_learning.marker == CellState.O:
                                who_is_learning.apply_gradient_by_episode(+1)
                            else:
                                who_is_learning.apply_gradient_by_episode(-1)

                            break

                        if state == BoardState.DRAW:
                            neural_net_player_agent1.draw_count += 1
                            neural_net_player_agent2.draw_count += 1

                            if who_is_learning.marker == CellState.O:
                                who_is_learning.apply_gradient_by_episode(+1)

                            if who_is_learning.marker == CellState.X:
                                who_is_learning.apply_gradient_by_episode(+0.1)

                            break

                    current_marker = board.another_marker(current_marker)

            who_is_learning.neural_net.save()

            print(
                '{:>6} learn={}  {} win_count={:<3} DRAW_count={:<3} {} win_count={:<3} (learning_rate={:.10f}) {}/{} {}/{}'.format(
                    episode_no,
                    board.markerToChar(who_is_learning.marker),
                    board.markerToChar(neural_net_player_agent1.marker),
                    neural_net_player_agent1.win_count,
                    neural_net_player_agent1.draw_count,
                    board.markerToChar(neural_net_player_agent2.marker),
                    neural_net_player_agent2.win_count,
                    learning_rate,

                    neural_net_player_agent1.move_count,
                    neural_net_player_agent1.random_move_count,
                    neural_net_player_agent2.move_count,
                    neural_net_player_agent2.random_move_count

                ))


def load_and_play():
    while True:
        print('-----start a game------')
        board = Board(in_dimension, in_win_row_len)

        human_player_agent = InputPlayerAgent(board, CellState.O)

        neural_net_player_agent = NeuralNetPlayerAgent(board, CellState.X)

        try:
            neural_net_player_agent.neural_net.load()
            print('loaded ' + in_save_file_name)
        except:
            pass

        play(board, neural_net_player_agent, human_player_agent)
        if human_player_agent.exit:
            return

        board.print()
        if human_player_agent.exit:
            return


if __name__ == "__main__":

    print("""
    Menu:
    1-train NN-vs-Bot and save the best neural network configuration {}x{} {}  
    2-train NN-vs-NN and save the best neural network configuration {}x{} {}
    3-load neural network configuration and play '{}'
    4-play with Bot agent
    """.format(in_dimension, in_dimension, in_win_row_len, in_dimension, in_dimension, in_win_row_len,
               in_save_file_name))

    choise = input(":")[0:1]

    if choise == '1':
        train_and_save_best_NNvsBot()

    if choise == '2':
        train_and_save_best_NNvsNN()

    if choise == '3':
        load_and_play()

    if choise == '4':
        board = Board(in_dimension, in_win_row_len)
        agent1 = InputPlayerAgent(board, CellState.X)
        agent2 = BotPlayerAgent(board, CellState.O)
        agent2.debug = True
        agent2.random_move_percent = 0.0
        while True:
            play(board, agent1, agent2)
            if agent1.exit or agent2.exit:
                break
