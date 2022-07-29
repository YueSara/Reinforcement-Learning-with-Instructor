import pickle


class Instructor:
    # the Instructor is a Q-Agent that was trained against a random player
    def __init__(self):
        self.t1 = 'policy_P1_vs_random'
        self.t2 = 'policy_P2_vs_random'
        self.teacher_policy = None

    def load_instructor(self, player):
        if player == 'X':
            with open(self.t1, "rb") as file:
                self.teacher_policy = pickle.load(file)

        elif player == 'O':
            with open(self.t2, "rb") as file:
                self.teacher_policy = pickle.load(file)

    def get_action(self, available_actions, state):
        # finds best action according to qtable
        maximum = float('-inf')
        best_action = 0
        for a in available_actions:
            value = self.get_q_teacher(a, state)
            if value > maximum:
                maximum = value
                best_action = a
        return best_action

    def get_q_teacher(self, action, state):
        state_action_key = state.get_state_str(action)
        if state_action_key not in self.teacher_policy:
            self.teacher_policy[state_action_key] = 0
        return self.teacher_policy.get(state_action_key)
