import torch
import numpy as np


class QL_Agent:
    """
    The constructor initializes the agent's state variables.
    state_a_dict is a dictionary that maps a state to a matrix of Q-values for actions,
    exploreP is the initial exploration probability,
    obs_vec is a vector of observations,
    gamma is the temporal discount factor,
    and alpha is the learning rate.
    """

    def __init__(self, env):
        """ Initializing the class"""
        self.state_a_dict = {}
        self.exploreP = 1
        self.obs_vec = []
        self.gamma = 0.99
        self.alpha = 0.01
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def obs_to_state(self, observation):
        """
        Converts an observation into a state index.
        If the observation is already present in obs_vec, returns the corresponding index.
        Otherwise, adds the observation to obs_vec and returns the new index.
        """
        state = -1
        for i in range(len(self.obs_vec)):
            if np.array_equal(observation, self.obs_vec[i]):
                state = i
                break
        if state == -1:
            state = len(self.obs_vec)
            self.obs_vec.append(observation)
        return state

    def take_action(self, state):
        """
        Selects an action based on the agent's policy (epsilon-greedy).
        If a random value is greater than exploreP (exploitation), the agent takes the action
        corresponding to the maximum Q-value.
        Otherwise (exploration), the agent takes a random action.
        The exploration probability exploreP decreases over time.
        """
        q_a = self.get_qval(state)
        if np.random.rand() > self.exploreP:
            """ exploitation"""
            action = np.argmax(q_a)
        else:
            """ exploration"""
            action = self.env.action_space.sample()
        self.exploreP *= 0.99
        return action

    def get_qval(self, state):
        """
        Returns the Q-values corresponding to a state.
        If the state is not present in state_a_dict, it randomly initializes the Q-values.
        """
        if state not in self.state_a_dict:
            # Initialize with random values, assuming action space size is 5 (for RIGHT_ONLY mapping)
            self.state_a_dict[state] = np.random.rand(self.env.action_space.n, 1) 
        return self.state_a_dict[state]

    def update_qval(self, action, state, reward, next_state, terminal):
        """
        Updates Q-values based on the Q-learning update equation:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        Calculates the temporal difference target (TD_target) and the temporal difference 
        error (td_error) to update the Q-value for the current state-action pair.
        """
        if terminal:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.amax(self.get_qval(next_state))

        td_error = td_target - self.get_qval(state)[action]
        self.state_a_dict[state][action] += self.alpha * td_error

    def update_qval_sarsa(self, action, state, reward, next_state, next_action, terminal):
        """
        Updates Q-values based on the SARSA update equation:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
        """
        if terminal:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.get_qval(next_state)[next_action]

        td_error = td_target - self.get_qval(state)[action]
        self.state_a_dict[state][action] += self.alpha * td_error