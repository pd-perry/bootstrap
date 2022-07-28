import numpy as np
import random
import torch

def CreateEnv(num_states, num_actions, R, seed):
    random.seed(seed)
    np.random.seed(seed)

    reward = np.random.randint(R, size=(num_states, num_actions))
    trans_p = np.zeros([num_states, num_actions, num_states])
    for i in range(num_states):
        for j in range(num_actions):
            sample = np.random.gamma(1, 1, num_states)
            trans_p[i, j, :] = sample / np.sum(sample)
    return reward, trans_p

def Sarsa(num_agent, num_states, num_actions, env_reward, env_trans_p, horizon):
    gamma = 0.9
    learning_rate = 0.1
    epsilon_greedy = 0.2

    time_step = 0
    q_values = np.random.random((num_states, num_actions, num_agents))
    state = np.random.randint(0, num_states, (1, num_agents))
    action = np.argmax(q_values[state])

    while time_step < horizon:
        reward = env_reward[state, action]
        next_state = np.random.choice(list(range(num_states)), p=env_trans_p[state, action].squeeze())

        explore = np.random.random(1) < epsilon_greedy
        if explore:
            next_action = np.random.randint(0, num_actions, 1)
        else:
            next_action = np.argmax(q_values[next_state])

        q_values[state, action] += learning_rate * (reward + gamma * q_values[next_state, next_action] - q_values[state, action])
        state = next_state
        action = next_action

        time_step += 1

    return q_values


if __name__ == "__main__":
    seed = 100
    num_states = 20
    num_actions = 10
    R = 5
    horizon = 200

    env_reward, env_trans_p = CreateEnv(num_states, num_actions, R, seed)

    num_agents = [1]

    for num_agent in num_agents:
        q_values = Sarsa(num_agent, num_states, num_actions, env_reward, env_trans_p, horizon)


    print(q_values)