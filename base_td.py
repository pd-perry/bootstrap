import numpy as np
import random
import torch

_device_ddtype_tensor_map = {
    'cpu': {
        torch.float32: torch.FloatTensor,
        torch.float64: torch.DoubleTensor,
        torch.float16: torch.HalfTensor,
        torch.uint8: torch.ByteTensor,
        torch.int8: torch.CharTensor,
        torch.int16: torch.ShortTensor,
        torch.int32: torch.IntTensor,
        torch.int64: torch.LongTensor,
        torch.bool: torch.BoolTensor,
    },
    'cuda': {
        torch.float32: torch.cuda.FloatTensor,
        torch.float64: torch.cuda.DoubleTensor,
        torch.float16: torch.cuda.HalfTensor,
        torch.uint8: torch.cuda.ByteTensor,
        torch.int8: torch.cuda.CharTensor,
        torch.int16: torch.cuda.ShortTensor,
        torch.int32: torch.cuda.IntTensor,
        torch.int64: torch.cuda.LongTensor,
        torch.bool: torch.cuda.BoolTensor,
    }
}

def CreateEnv(num_states, num_actions, R, seed):
    random.seed(seed)
    np.random.seed(seed)

    reward = torch.abs(torch.randn(num_states, num_actions))
    dist_trans_p = torch.distributions.dirichlet.Dirichlet(torch.ones(num_states))
    trans_p = dist_trans_p.sample([num_states, num_actions])
    return reward, trans_p

def Sarsa(num_agent, num_states, num_actions, env_reward, env_trans_p, horizon, upper_confidence=False):
    gamma = 0.9
    learning_rate = 0.1
    epsilon_greedy = 0.2

    num_selected = torch.ones(num_agent)

    time_step = 1
    q_values = torch.rand((num_states, num_actions, num_agent))
    state = torch.randint(0, num_states, (num_agent, ))
    action = torch.argmax(q_values[state, :, torch.arange(num_agent)], -1)

    while time_step <= horizon:
        reward = env_reward[state, action]
        # next_state = torch.tensor([np.random.choice(list(range(num_states)), p=env_trans_p[state[i], action[i]].squeeze()) for i in range(num_agent)])
        next_state = torch.distributions.categorical.Categorical(env_trans_p[state, action]).sample()
        if upper_confidence:
            upper_confidence_q = q_values.clone() + torch.sqrt(torch.div(torch.full_like(num_selected, np.log(time_step)), num_selected))
            next_action = torch.argmax(upper_confidence_q[next_state, :, torch.arange(num_agent)], -1)
        else:
            explore = np.random.random(1) < epsilon_greedy
            if explore:
                next_action = np.random.randint(0, num_actions, 1)
            else:
                next_action = np.argmax(q_values[next_state])

        q_values[state, action, torch.arange(num_agent)] += learning_rate * (reward + gamma * q_values[next_state, next_action, torch.arange(num_agent)] - q_values[state, action, torch.arange(num_agent)])
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

    num_agents = [3]

    for num_agent in num_agents:
        q_values = Sarsa(num_agent, num_states, num_actions, env_reward, env_trans_p, horizon, True)


    print(q_values)