import numpy as np
import random
import pickle
import sys


class Player:
    def __init__(self):
        self.tag = ''


class RandomPlayer(Player):
    def __init__(self):
        super().__init__()

    def make_move(self, available_actions, state, updateQ):
        chosen_action = random.choice(available_actions)
        state.place_tag(chosen_action, self.tag)


class QAgent(Player):
    def __init__(self):
        super().__init__()
        self.epsilon = 1.0  # at the beginning 100% random actions, reduced with every round
        self.alpha = 0.2
        self.gamma = 0.9
        self.qtable = dict()  # key: state-action-string

    def reduce_eps(self, rounds):
        # reduce self.epsilon every round
        self.epsilon = max(0.1, self.epsilon - 1 / rounds)

    def make_move(self, available_actions, state, updateQ):
        if updateQ:
            if np.random.uniform(0, 1) < self.epsilon:
                chosen_action = random.choice(available_actions)
                self.updateQ(chosen_action, state)
            else:
                chosen_action = self.optimal_action(available_actions, state)
                self.updateQ(chosen_action, state)
        else:
            chosen_action = self.optimal_action(available_actions, state)
            state.place_tag(chosen_action, self.tag)

    def get_teacher_move(self, available_actions, state, instructor):
        chosen_action = instructor.get_action(available_actions, state)
        self.updateQ(chosen_action, state)

    def check_strategy(self, available_actions, state, updateQ, steps, sum_steps, strategy, instructor):
        if not updateQ:
            # an agent can only use a teacher during training, if updateQ is set to False, stop the game
            print('The Player uses an Instructor but is not training! Start a new game with correct update values.')
            sys.exit()
        else:
            if strategy == 1:
                # every first move (for both players) done by instructor
                if steps == 0 or steps == 1:
                    self.get_teacher_move(available_actions, state, instructor)
                else:
                    self.make_move(available_actions, state, updateQ)

            elif strategy == 2:
                # every 20 steps (5%) done by instructor
                if sum_steps % 20 == 0:
                    self.get_teacher_move(available_actions, state, instructor)
                else:
                    self.make_move(available_actions, state, updateQ)

            elif strategy == 3:
                # every 5 steps (20%) done by instructor
                if sum_steps % 5 == 0:
                    self.get_teacher_move(available_actions, state, instructor)
                else:
                    self.make_move(available_actions, state, updateQ)
            else:
                correct_strategy = 0
                while correct_strategy not in range(1, 4):
                    try:
                        correct_strategy = int(input('Invalid strategy number, choose 1, 2, or 3: '))
                    except ValueError:
                        print('Enter a valid integer')
                self.check_strategy(available_actions, state, updateQ, steps, sum_steps, correct_strategy, instructor)

    def optimal_action(self, available_actions, state):
        # finds best action according to qtable
        maximum = float('-inf')
        best_action = 0
        for a in available_actions:
            value = self.get_q(a, state)
            if value > maximum:
                maximum = value
                best_action = a
        return best_action

    # reward system for 1st player: focusses on wins rather than draws
    # reward system for 2nd player: focusses on draws rather than wins
    def reward(self, state):
        _, winner = state.check_winner()
        reward = 0
        # if action leads to 1st or 2nd player win: reward = 1 or 0.5
        if winner == self.tag:
            if self.tag == "X":
                reward = 1
            elif self.tag == "O":
                reward = 0.5
        # if action leads to draw: reward = 0.5
        # 2nd player can't make last move, last action in a draw will always be by X
        elif winner == "draw":
            reward = 0.5
        # if 2nd player action leads to draw (X only has one action option left, which leads to draw): reward = 1
        elif self.tag == "O" and len(state.available_actions()) == 1:
            a = state.available_actions()
            state.place_tag(a, "X")  # simulate action
            _, w = state.check_winner()
            state.place_tag(a, 0)  # undo simulated action
            if w == "draw":
                reward = 1
        # if action potentially leads to loss: reward = -1
        elif state.loss_possible(self.tag):
            reward = -1
        # action that doesn't cause win, draw, or loss
        else:
            reward = 0
        return reward

    def get_q(self, action, state):
        state_action_key = state.get_state_str(action)
        if state_action_key not in self.qtable:
            self.qtable[state_action_key] = 0
        return self.qtable.get(state_action_key)

    def updateQ(self, action, state):
        if self.tag == 'X':
            opponent = 'O'
        else:
            opponent = 'X'

        # get current Q-value and key
        Q = self.get_q(action, state)
        state_action_key = state.get_state_str(action)

        # perform the action in that state, produce new state
        state.place_tag(action, self.tag)
        # now opponent's turn
        # simulate opponent's best action in the new state (while using own qtable)
        # for this, switch X and O
        opponent_actions = state.available_actions()
        max_next_Q = float('-inf')
        if len(opponent_actions) != 0:
            state.switch_player()
            # find opponent's best action
            best_action = self.optimal_action(opponent_actions, state)
            # switch back to normal
            state.switch_player()
            # then place opponent's tag into normal state
            state.place_tag(best_action, opponent)
            # now agent's turn again, get all possible next actions & their maxQ
            my_next_actions = state.available_actions()
            if len(my_next_actions) != 0:
                for a in my_next_actions:
                    value = self.get_q(a, state)
                    if value > max_next_Q:
                        max_next_Q = value
            else:
                max_next_Q = 0
            # undo simulated action
            state.place_tag(best_action, 0)

        else:  # if len(opponent_actions) == 0
            max_next_Q = 0

        new_Q = (1 - self.alpha) * Q + self.alpha * (self.reward(state) + self.gamma * max_next_Q)
        self.qtable[state_action_key] = new_Q

    def save_policy(self, path):
        # always save at the end, because even when updateQ = False, if a key is not existing it is updated to 0
        with open(path, "wb") as file:
            pickle.dump(self.qtable, file)

    def load_policy(self, path):
        with open(path, "rb") as file:
            self.qtable = pickle.load(file)
