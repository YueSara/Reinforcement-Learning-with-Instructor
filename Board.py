import numpy as np


class Board:
    def __init__(self):
        self.state = np.zeros(9, dtype=object)

    def print_game(self):
        s = self.state
        print('    {} | {} | {}'.format(s[0], s[1], s[2]))
        print('  --------------')
        print('    {} | {} | {}'.format(s[3], s[4], s[5]))
        print('  --------------')
        print('    {} | {} | {}'.format(s[6], s[7], s[8]))
        print('  --------------')
        print('  --------------')

    def check_winner(self):
        # define winning positions
        board = self.state.reshape(3, 3)
        row1 = board[0]
        row2 = board[1]
        row3 = board[2]

        board_transposed = board.transpose()
        col1 = board_transposed[0]
        col2 = board_transposed[1]
        col3 = board_transposed[2]

        dia1 = board.diagonal()
        dia2 = np.fliplr(board).diagonal()

        winning_positions = np.array([row1, row2, row3, col1, col2, col3, dia1, dia2])

        for pos in winning_positions:
            # if first element in line is the same as other two elements in that line AND is not zero, there is a winner
            if np.all(pos == pos[0]) and pos[0] != 0:
                game_over = True
                if pos[0] == 'X':
                    winner = 'X'
                else:
                    winner = 'O'
                return game_over, winner
        # if there was no winner but no moves are available, it is a draw, else the game is not over yet
        if len(self.available_actions()) == 0:
            game_over = True
            winner = 'draw'
        else:
            game_over = False
            winner = None
        return game_over, winner

    def available_actions(self):
        # returns list of indeces of possible actions by filtering positions without tags
        available_actions = [idx for idx, val in enumerate(self.state) if (val == 0)]
        return available_actions

    def loss_possible(self, tag):
        # checks if action enables opponent to win, i.e. if agent missed an opportunity to prevent opponent's win
        # simulates all possible opponent actions, then checks winner
        # if opponent won, loss is possible
        if tag == 'X':
            opponent = 'O'
        else:
            opponent = 'X'
        loss = False
        for a in self.available_actions():
            self.place_tag(a, opponent)
            _, winner = self.check_winner()
            self.place_tag(a, 0)
            if winner == opponent:
                loss = True
                break
        return loss

    def switch_player(self):
        # replace X and O positions
        self.state[self.state == 'X'] = 'tmpX'
        self.state[self.state == 'O'] = 'X'
        self.state[self.state == 'tmpX'] = 'O'

    def place_tag(self, chosen_action, tag):
        self.state[chosen_action] = tag

    def get_state_str(self, action):
        state_action_key = str(self.state) + str(action)
        return state_action_key

    def reset(self):
        self.state = np.zeros(9, dtype=object)
