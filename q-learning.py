#-------------------------------------------------------------------------------
# A Python implementation of Q-Learning for path finding problem [2].
#
# Author: Tung D. Le
#
# This implementation is built on top of the following works:
# [1] Original Python implementation: http://firsttimeprogrammer.blogspot.com/2016/09/getting-ai-smarter-with-q-learning.html
# [2] A painless Q-Learning tutorial: http://mnemstudio.org/path-finding-q-learning-tutorial.htm
# [3] A brief introduction to reinforcement learning: https://www.cs.ubc.ca/~murphyk/Bayes/pomdp.html.
#
# Different from [1], we re-organize the code where Agent and Environment are
# introduced, according to their definitions in [3]. This helps understand
# reinforcement learning in a more intuitive way.
#-------------------------------------------------------------------------------

import numpy as np

def argmax(x):
    max_index = np.where(x == np.max(x))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    return max_index

#-------------------------------------------------------------------------------
# Environment = (S, A, t, o, r)
# S: a finite set of states
# A: a finite set of actions
# t: state transition function (state -> action -> state)
# o: observation function (state -> action -> state)
# r: reward function (state -> action -> reward)
class Environment:
    def __init__(self, S, A):
        self.S = S
        self.A = A
        # Represent the reward function by a state-action matrix.
        self.R = None

    # prev_state -> action -> state
    def transition(self, prev_state, action):
        if self.valid_action(prev_state, action):
            new_state = action
        else:
            new_state = prev_state
        return new_state

    # prev_state -> action -> observation
    def observation(self, prev_state, action):
        # the world is fully observable
        return self.transition(prev_state, action)

    # prev_state -> action -> reward
    def reward(self, prev_state, action):
        if self.valid_action(prev_state, action):
            return self.R[prev_state, action]
        else:
            return 0

    # helper function
    def valid_action(self, prev_state, action):
        return self.R[prev_state, action] >= 0

    # helper function
    def set_reward_matrix(self, R):
        self.R = R


#-------------------------------------------------------------------------------
# Agent = (S, A, t, p)
# S: a finite set of states
# A: a finite set of actions
# t: state transition function (state -> action -> observation -> reward -> new_state)
# p: policy function (state -> action)
class Agent:
    def __init__(self, S, A):
        self.S = S
        self.A = A

    # prev_state -> action -> observation -> reward -> state
    def transition(self, prev_state, action, observation, reward):
        # the world is fully observable
        return observation

    # state -> action
    # The optimial policy can be found by maximising over Q*(s, a).
    def policy(self, state, Q):
        return argmax(Q[state,])


#-------------------------------------------------------------------------------
# Training

state_space = np.arange(0, 6)
action_space = np.arange(0, 6)
terminal_state = 5

# Input graph: Node 5 is an absorbing state.
#
#                  +-------+
#                  |   1   |--------------+
#                  +---+---+              |
#                      |                  |
#                      |                  |  +----+
#                      |                  |  |    |
# +------+         +-------+           +-------+  |
# |  2   |---------|   3   |           |   5   |  |
# +------+         +-------+           +--+----+  |
#                      |                  |  |    |
#                      |                  |  +----+
#                      |                  |
# +------+         +---+---+              |
# |  0   |---------|   4   |--------------+
# +------+         +-------+

# Environment
env = Environment(state_space, action_space)
# Immediate rewards
# - -1: invalid action.
# - Get 100 if reaching the terminal state. Otherwise, 0.
R = np.matrix([[-1, -1, -1, -1, 0, -1],
               [-1, -1, -1, 0, -1, 100],
               [-1, -1, -1, 0, -1, -1],
               [-1, 0, 0, -1, 0, -1],
               [0, -1, -1, 0, -1, 100],
               [-1, 0, -1, -1, 0, 100]])
env.set_reward_matrix(R)


# Agent
agent = Agent(state_space, action_space)

# Discount factor
gamma = 0.8

# Train to approximate Q(s, a).
# Initialize Q(s,a)
Q = np.matrix(np.zeros([len(state_space), len(action_space)]))
for i in range(200000):
    # 1. Initialize a state.
    random_index = np.random.randint(0, len(state_space))
    prev_state = state_space[random_index] 

    # 2.Do one episode.
    while True: 
        # 2.1. Choose an action using policy derived from Q.
        action = agent.policy(prev_state, Q)

        # 2.2. Take the action, and observe reward.
        _ = env.transition(prev_state, action)
        # Get feedback from environment
        observation = env.observation(prev_state, action)
        immediate_reward = env.reward(prev_state, action)

        # 2.3. Update Q(s, a) using Bellman Optimiality Equation for Q*(s,a).
        # Optimal future reward
        state = agent.transition(
            prev_state, action, observation, immediate_reward)
        future_action = argmax(Q[state,])
        future_reward = gamma * Q[state, future_action]
        # Update the discounted reward
        Q[prev_state, action] = immediate_reward + future_reward
        
        # 2.4. Terminate if coming to the terminal state. 
        if (state == terminal_state):
            break
        else:
            prev_state = state

# Print the agent's learned rewards
print("Agent's discounted rewards (Q matrix, normalized):")
print(Q / np.max(Q) * 100)

#-------------------------------------------------------------------------------
# Testing

# Find the shortest path from 2 to 5
# Best sequence path starting from 2: [2, 3, 1, 5] or [2, 3, 4, 5]
starting_state = 2
target_state = 5
steps = [starting_state]
while (starting_state != target_state):
    # Select the best action
    starting_action = agent.policy(starting_state, Q)
    # Move to the next state
    next_step = env.transition(starting_state, starting_action)
    # Store intermediate results
    steps.append(next_step)
    # Reset
    starting_state = next_step

# Print selected sequence of steps
print("Selected path: ", steps)

#-------------------------------------------------------------------------
#                               OUTPUT
#-------------------------------------------------------------------------
#
# Agent's discounted rewards (Q matrix, normalized):
# [[  0.   0.   0.   0.  80.   0.]
#  [  0.   0.   0.   0.   0. 100.]
#  [  0.   0.   0.  64.   0.   0.]
#  [  0.  80.   0.   0.   0.   0.]
#  [  0.   0.   0.   0.   0. 100.]
#  [  0.   0.   0.   0.   0. 100.]]
# ('Selected path: ', [2, 3, 1, 5])
