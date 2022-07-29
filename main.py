from Game import Game
from Player import RandomPlayer, QAgent

'''BASELINE'''


def random_random():
    print('Random Player vs. Random Player')
    game = Game(RandomPlayer(), RandomPlayer(),
                policy_p1=None, policy_p2=None,
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


'''TRAINING'''


def agent_random_train():
    print('Q-Agent vs. Random Player | Training')
    game = Game(QAgent(), RandomPlayer(),
                policy_p1="policy_P1_vs_random", policy_p2=None,
                updateQ1=True, updateQ2=False)
    game.play()
    game.show_results()


def random_agent_train():
    print('Random Player vs. Q-Agent | Training')
    game = Game(RandomPlayer(), QAgent(),
                policy_p1=None, policy_p2="policy_P2_vs_random",
                updateQ1=False, updateQ2=True)
    game.play()
    game.show_results()


def agent_agent_train():
    print('Q-Agent vs. Q-Agent | Training')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_qvq", policy_p2="policy_P2_qvq",
                updateQ1=True, updateQ2=True)
    game.play()
    game.show_results()


def agent_trained_train():
    print('Q-Agent vs. trained Q-Agent | Training')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_vs_trained", policy_p2="policy_P2_qvq",
                updateQ1=True, updateQ2=False)
    game.play()
    game.show_results()


def trained_agent_train():
    print('Trained Q-Agent vs. Q-Agent | Training')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_qvq", policy_p2="policy_P2_vs_trained",
                updateQ1=False, updateQ2=True)
    game.play()
    game.show_results()


'''TESTING'''


def agent_random_test():
    print('Q-Agent vs. Random Player | Testing')
    game = Game(QAgent(), RandomPlayer(),
                policy_p1="policy_P1_vs_random", policy_p2=None,
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


def random_agent_test():
    print('Random Player vs. Q-Agent | Testing')
    game = Game(RandomPlayer(), QAgent(),
                policy_p1=None, policy_p2="policy_P2_vs_random",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


def agent_agent_test():
    print('Q-Agent vs. Q-Agent | Testing')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_qvq", policy_p2="policy_P2_qvq",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


def agent_trained_test():
    print('Q-Agent vs. trained Q-Agent | Testing')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_vs_trained", policy_p2="policy_P2_qvq",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


def trained_agent_test():
    print('Trained Q-Agent vs. Q-Agent | Testing')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_qvq", policy_p2="policy_P2_vs_trained",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


# only for testing 
def trained_trained_test():
    print('Trained Q-Agent vs. Trained Q-Agent | Testing')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_vs_trained", policy_p2="policy_P2_vs_trained",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


########################################################################################################################
'''TRAINING WITH INSTRUCTOR'''

'''P1 vs. Random Player'''


# 1.1 every first step
def teacher1_random_train():
    print('Q-Agent with Instructor (strategy 1) vs. Random Player | Training')
    game = Game(QAgent(), RandomPlayer(),
                policy_p1="policy_P1_vs_random_WITH_TEACHER_X_1", policy_p2=None,
                updateQ1=True, updateQ2=False, teacher='X', strategy=1)
    game.play()
    game.show_results()


# 1.2 5% of all steps (every 20 steps)
def teacher2_random_train():
    print('Q-Agent with Instructor (strategy 2) vs. Random Player | Training')
    game = Game(QAgent(), RandomPlayer(),
                policy_p1="policy_P1_vs_random_WITH_TEACHER_X_2", policy_p2=None,
                updateQ1=True, updateQ2=False, teacher='X', strategy=2)
    game.play()
    game.show_results()


# 1.3 20% of all steps (every 5 steps)
def teacher3_random_train():
    print('Q-Agent with Instructor (strategy 3) vs. Random Player | Training')
    game = Game(QAgent(), RandomPlayer(),
                policy_p1="policy_P1_vs_random_WITH_TEACHER_X_3", policy_p2=None,
                updateQ1=True, updateQ2=False, teacher='X', strategy=3)
    game.play()
    game.show_results()


'''P2 vs. Random Player'''


# 2.1 every first step
def random_teacher1_train():
    print('Random Player vs. Q-Agent with Instructor (strategy 1) | Training')
    game = Game(RandomPlayer(), QAgent(),
                policy_p1=None, policy_p2="policy_P2_vs_random_WITH_TEACHER_O_1",
                updateQ1=False, updateQ2=True, teacher='O', strategy=1)
    game.play()
    game.show_results()


# 2.2 5% of all steps (every 20 steps)
def random_teacher2_train():
    print('Random Player vs. Q-Agent with Instructor (strategy 2) | Training')
    game = Game(RandomPlayer(), QAgent(),
                policy_p1=None, policy_p2="policy_P2_vs_random_WITH_TEACHER_O_2",
                updateQ1=False, updateQ2=True, teacher='O', strategy=2)
    game.play()
    game.show_results()


# 2.3 20% of all steps (every 5 steps)
def random_teacher3_train():
    print('Random Player vs. Q-Agent with Instructor (strategy 3) | Training')
    game = Game(RandomPlayer(), QAgent(),
                policy_p1=None, policy_p2="policy_P2_vs_random_WITH_TEACHER_O_3",
                updateQ1=False, updateQ2=True, teacher='O', strategy=3)
    game.play()
    game.show_results()


'''P1 vs. Trained Agent'''


# 3.1 every first step
def teacher1_trained_train():
    print('Q-Agent with Instructor (strategy 1) vs. Trained Q-Agent | Training')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_vs_trained_WITH_TEACHER_X_1", policy_p2="policy_P2_qvq",
                updateQ1=True, updateQ2=False, teacher='X', strategy=1)
    game.play()
    game.show_results()


# 3.2 5% of all steps (every 20 steps)
def teacher2_trained_train():
    print('Q-Agent with Instructor (strategy 2) vs. Trained Q-Agent | Training')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_vs_trained_WITH_TEACHER_X_2", policy_p2="policy_P2_qvq",
                updateQ1=True, updateQ2=False, teacher='X', strategy=2)
    game.play()
    game.show_results()


# 3.3 20% of all steps (every 5 steps)
def teacher3_trained_train():
    print('Q-Agent with Instructor (strategy 3) vs. Trained Q-Agent | Training')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_vs_trained_WITH_TEACHER_X_3", policy_p2="policy_P2_qvq",
                updateQ1=True, updateQ2=False, teacher='X', strategy=3)
    game.play()
    game.show_results()


'''P2 vs. Trained Agent'''


# 4.1 every first step
def trained_teacher1_train():
    print('Trained Q-Agent vs. Q-Agent with Instructor (strategy 1) | Training')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_qvq", policy_p2="policy_P2_vs_trained_WITH_TEACHER_O_1",
                updateQ1=False, updateQ2=True, teacher='O', strategy=1)
    game.play()
    game.show_results()


# 4.2 5% of all steps (every 20 steps)
def trained_teacher2_train():
    print('Trained Q-Agent vs. Q-Agent with Instructor (strategy 2) | Training')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_qvq", policy_p2="policy_P2_vs_trained_WITH_TEACHER_O_2",
                updateQ1=False, updateQ2=True, teacher='O', strategy=2)
    game.play()
    game.show_results()


# 4.3 20% of all steps (every 5 steps)
def trained_teacher3_train():
    print('Trained Q-Agent vs. Q-Agent with Instructor (strategy 3) | Training')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_qvq", policy_p2="policy_P2_vs_trained_WITH_TEACHER_O_3",
                updateQ1=False, updateQ2=True, teacher='O', strategy=3)
    game.play()
    game.show_results()


'''TESTING WITH INSTRUCTOR'''

'''P1 vs. Random Player'''


# 1.1 every first step
def teacher1_random_test():
    print('Q-Agent with Instructor (strategy 1) vs. Random Player | Testing')
    game = Game(QAgent(), RandomPlayer(),
                policy_p1="policy_P1_vs_random_WITH_TEACHER_X_1", policy_p2=None,
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


# 1.2 5% of all steps (every 20 steps)
def teacher2_random_test():
    print('Q-Agent with Instructor (strategy 2) vs. Random Player | Testing')
    game = Game(QAgent(), RandomPlayer(),
                policy_p1="policy_P1_vs_random_WITH_TEACHER_X_2", policy_p2=None,
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


# 1.3 20% of all steps (every 5 steps)
def teacher3_random_test():
    print('Q-Agent with Instructor (strategy 3) vs. Random Player | Testing')
    game = Game(QAgent(), RandomPlayer(),
                policy_p1="policy_P1_vs_random_WITH_TEACHER_X_3", policy_p2=None,
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


'''P2 vs. Random Player'''


# 2.1 every first step
def random_teacher1_test():
    print('Random Player vs. Q-Agent with Instructor (strategy 1) | Testing')
    game = Game(RandomPlayer(), QAgent(),
                policy_p1=None, policy_p2="policy_P2_vs_random_WITH_TEACHER_O_1",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


# 2.2 5% of all steps (every 20 steps)
def random_teacher2_test():
    print('Random Player vs. Q-Agent with Instructor (strategy 2) | Testing')
    game = Game(RandomPlayer(), QAgent(),
                policy_p1=None, policy_p2="policy_P2_vs_random_WITH_TEACHER_O_2",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


# 2.3 20% of all steps (every 5 steps)
def random_teacher3_test():
    print('Random Player vs. Q-Agent with Instructor (strategy 3) | Testing')
    game = Game(RandomPlayer(), QAgent(),
                policy_p1=None, policy_p2="policy_P2_vs_random_WITH_TEACHER_O_3",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


'''P1 vs. Trained Agent'''


# 3.1 every first step
def teacher1_trained_test():
    print('Q-Agent with Instructor (strategy 1) vs. Trained Q-Agent | Testing')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_vs_trained_WITH_TEACHER_X_1", policy_p2="policy_P2_qvq",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


# 3.2 5% of all steps (every 20 steps)
def teacher2_trained_test():
    print('Q-Agent with Instructor (strategy 2) vs. Trained Q-Agent | Testing')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_vs_trained_WITH_TEACHER_X_2", policy_p2="policy_P2_qvq",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


# 3.3 20% of all steps (every 5 steps)
def teacher3_trained_test():
    print('Q-Agent with Instructor (strategy 3) vs. Trained Q-Agent | Testing')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_vs_trained_WITH_TEACHER_X_3", policy_p2="policy_P2_qvq",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


'''P2 vs. Trained Agent'''


# 4.1 every first step
def trained_teacher1_test():
    print('Trained Q-Agent vs. Q-Agent with Instructor (strategy 1) | Testing')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_qvq", policy_p2="policy_P2_vs_trained_WITH_TEACHER_O_1",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


# 4.2 5% of all steps (every 20 steps)
def trained_teacher2_test():
    print('Trained Q-Agent vs. Q-Agent with Instructor (strategy 2) | Testing')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_qvq", policy_p2="policy_P2_vs_trained_WITH_TEACHER_O_2",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


# 4.3 20% of all steps (every 5 steps)
def trained_teacher3_test():
    print('Trained Q-Agent vs. Q-Agent with Instructor (strategy 3) | Testing')
    game = Game(QAgent(), QAgent(),
                policy_p1="policy_P1_qvq", policy_p2="policy_P2_vs_trained_WITH_TEACHER_O_3",
                updateQ1=False, updateQ2=False)
    game.play()
    game.show_results()


########################################################################################################################
'''Baseline'''
# random_random()

'''vs Random'''
# agent_random_train()
# agent_random_test()

# random_agent_train()
# random_agent_test()

'''vs Agent'''
# agent_agent_train()
# agent_agent_test()

'''vs Trained Agent'''
# agent_trained_train()
# agent_trained_test()

# trained_agent_train()
# trained_agent_test()

# trained_trained_test()

##################################################
'''with Instructor'''

'''vs Random'''
'''P1'''
teacher1_random_train()
teacher1_random_test()

# teacher2_random_train()
# teacher2_random_test()

# teacher3_random_train()
# teacher3_random_test()

'''P2'''
# random_teacher1_train()
# random_teacher1_test()

# random_teacher2_train()
# random_teacher2_test()

# random_teacher3_train()
# random_teacher3_test()

'''vs Trained Agent'''
'''P1'''
# teacher1_trained_train()
# teacher1_trained_test()

# teacher2_trained_train()
# teacher2_trained_test()

# teacher3_trained_train()
# teacher3_trained_test()

'''P2'''
# trained_teacher1_train()
# trained_teacher1_test()

# trained_teacher2_train()
# trained_teacher2_test()

# trained_teacher3_train()
# trained_teacher3_test()
