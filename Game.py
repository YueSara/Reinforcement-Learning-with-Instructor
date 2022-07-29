from Board import Board
from Player import QAgent
from Instructor import Instructor

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm


class Game:
    def __init__(self, player1, player2, policy_p1, policy_p2, updateQ1, updateQ2, teacher=None, strategy=None):
        self.state = Board()
        self.p1 = player1
        self.p2 = player2
        self.p1.tag = 'X'
        self.p2.tag = 'O'
        self.current_player = self.p1

        self.rounds = 100000
        self.pol1 = policy_p1
        self.pol2 = policy_p2
        self.updateQ1 = updateQ1
        self.updateQ2 = updateQ2

        self.teacher = teacher
        self.strategy = strategy

        # counter for all steps over all rounds, used to check strategy conditions
        self.sum_steps = 1
        # counter for number of steps per round
        self.steps = 0
        # counter lists
        self.steps_per_game = []
        self. winners_list = []

    def play(self):
        # if QAgents are playing (not training), policy is loaded
        if not self.updateQ1 and isinstance(self.p1, QAgent):
            self.p1.load_policy(self.pol1)
        if not self.updateQ2 and isinstance(self.p2, QAgent):
            self.p2.load_policy(self.pol2)

        # if an instructor is employed, load their policy
        if self.teacher is not None:
            instructor = Instructor()
            instructor.load_instructor(self.teacher)

        for i in tqdm(range(self.rounds)):
            game_over = False
            # X always starts
            self.current_player = self.p1
            training = self.updateQ1
            while not game_over:
                actions = self.state.available_actions()

                if self.current_player.tag == self.teacher:
                    self.current_player.check_strategy(
                        actions, self.state, training, self.steps, self.sum_steps, self.strategy, instructor
                    )
                else:
                    self.current_player.make_move(actions, self.state, training)
                # self.state.print_game()

                # count number of actions over all rounds
                self.sum_steps += 1
                # count number of actions taken per game
                self.steps += 1
                game_over, winner = self.state.check_winner()
                # log winner
                if winner is not None:
                    if winner == self.p1.tag:
                        self.winners_list.append(1)
                    elif winner == self.p2.tag:
                        self.winners_list.append(-1)
                    elif winner == "draw":
                        self.winners_list.append(0)
                # switch players
                if self.current_player == self.p1:
                    self.current_player = self.p2
                    training = self.updateQ2
                else:
                    self.current_player = self.p1
                    training = self.updateQ1

            self.steps_per_game.append(self.steps)
            # set steps to zero for new round
            self.steps = 0
            self.state.reset()
            if self.updateQ1 and isinstance(self.p1, QAgent):
                self.p1.reduce_eps(self.rounds)
            if self.updateQ2 and isinstance(self.p2, QAgent):
                self.p2.reduce_eps(self.rounds)

        # save policy after all rounds are done during training
        if self.updateQ1 and isinstance(self.p1, QAgent):
            self.p1.save_policy(self.pol1)
        if self.updateQ2 and isinstance(self.p2, QAgent):
            self.p2.save_policy(self.pol2)

    def moving_avg(self, window, results, winner_value=None):
        averages = []
        for i in range(window, len(results)+1):
            if winner_value is not None:
                count = len([x for x in results[i-window:i] if x == winner_value])
            else:
                count = sum([x for x in results[i-window:i]])
            averages.append(count/window)
        return averages

    def show_results(self):
        window = 1000
        p1_wins = [x for x in self.winners_list if x == 1]
        p2_wins = [x for x in self.winners_list if x == -1]
        draws = [x for x in self.winners_list if x == 0]
        print(f'Total Number of Games = {self.rounds}   ||   Win Percentage \n'
              '-------------------------------------------------\n'
              f'Player 1 Wins = {len(p1_wins)}            ||   {round(len(p1_wins)/self.rounds*100, 2)}% \n'
              f'Player 2 Wins = {len(p2_wins)}            ||   {round(len(p2_wins)/self.rounds*100, 2)}% \n'
              f'Draws = {len(draws)}                    ||   {round(len(draws)/self.rounds*100, 2)}%\n')
        p1_avg = self.moving_avg(window, self.winners_list, 1)
        p2_avg = self.moving_avg(window, self.winners_list, -1)
        draw_avg = self.moving_avg(window, self.winners_list, 0)
        x_values = range(window, self.rounds+1)
        plt.plot(x_values, p1_avg, color='red', label='Player 1')
        plt.plot(x_values, p2_avg, color='blue', label='Player 2')
        plt.plot(x_values, draw_avg, color='grey', label='Draws')
        plt.ylabel(f'Win Percentage per {window} Games')
        plt.xlabel('Number of Games')
        plt.title(f"Moving Average of Wins over {self.rounds} Games")
        plt.legend(loc='best')
        plt.xlim([0, self.rounds])
        plt.ylim([0, 1])
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        plt.show()

        print(f'Average of {sum(self.steps_per_game)/self.rounds} steps per game')
        steps_avg = self.moving_avg(window, self.steps_per_game)
        plt.plot(x_values, steps_avg, color='green')
        plt.ylabel(f'Avg. Steps per {window} Games')
        plt.xlabel('Number of Games')
        plt.title(f"Moving Average of Steps over {self.rounds} Games")
        plt.xlim([0, self.rounds])
        plt.ylim([5, 9])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        plt.show()
