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

def create_env(num_env, num_states, num_actions, R, seed):
    reward = torch.abs(torch.randn(num_env, num_states, num_actions))
    dist_trans_p = torch.distributions.dirichlet.Dirichlet(torch.ones(num_states))
    trans_p = dist_trans_p.sample([num_env, num_states, num_actions])
    return reward, trans_p

def format_experience(state, action, reward, next_state):
    experience = torch.vstack((state, action, reward, next_state))

    return experience.transpose(0, 1).tolist()

def select_action(q_values, state, num_selected=None, time_step=None, epsilon_greedy=0.2, upper_confidence=False):
    if upper_confidence:
        upper_confidence_q = q_values[env].clone() + \
                             0.5 * torch.sqrt(torch.div(torch.full_like(num_selected[env], np.log(time_step[env])),
                                                        num_selected[env])).unsqueeze(0).unsqueeze(-1).repeat(
            num_states, 1, num_agent)
        # print(0.5 * torch.sqrt(torch.div(torch.full_like(num_selected[env], np.log(time_step[env])), num_selected[env])).unsqueeze(0).unsqueeze(-1).repeat(num_states, 1, num_agent))
        action = torch.argmax(upper_confidence_q[state, :, torch.arange(num_agent)], -1)
    else:
        explore = torch.rand(1) < epsilon_greedy
        if explore:
            action = torch.randint(0, num_actions, (num_agent,), dtype=int)
        else:
            action = torch.argmax(q_values[env, state, :, torch.arange(num_agent)], -1)

    return action

def update_with_experience(q_values, buffer, learning_rate, gamma):
    for env in range(num_envs):
        # for exp in buffer[env]:
        #     s, a, r, s_p = exp
        exp = torch.tensor(buffer[env])
        state, action, reward, next_state = exp[:, 0].long(), exp[:, 1].long(), exp[:, 2], exp[:, 3].long()
        for agent in range(num_agent):
            q_values[env, state, action, agent] += learning_rate * (reward + gamma * torch.max(q_values[env, next_state, :, agent], -1)[0] - q_values[env, state, action, agent])
            # for i in range(len(state)):
            #     q_values[env, state[i], action[i], agent] += \
            #         learning_rate * (reward[i] + gamma * torch.max(q_values[env, next_state[i], :, agent], -1)[0] - q_values[env, state[i], action[i], agent])
    return q_values

def off_policy_td(num_env, num_episodes, num_agent, num_states, num_actions, env_reward, env_trans_p, all_env_optimal_policy, horizon, upper_confidence=False):
    gamma = 0.9
    learning_rate = 0.1
    epsilon_greedy = 0.2

    buffer = [[] for _ in range(num_envs)]
    num_selected = torch.ones((num_env, num_actions))
    q_values = torch.rand((num_env, num_states, num_actions, num_agent))
    # q_values = torch.rand((num_env, num_states, num_actions)).unsqueeze(-1).repeat(1, 1, 1, num_agent)
    time_step = torch.ones((num_env))
    cumulative_regret = torch.zeros((num_env, num_agent))

    random_track = torch.zeros((num_env, num_agent))
    optimal_track = torch.zeros((num_env, num_agent))
    td_track = torch.zeros((num_env, num_agent))
    for i in range(num_episodes):
        for env in range(num_env):
            state = torch.randint(0, num_states, (num_agent,))
            optimal_state = state
            random_state = state

            episodic_regret = torch.zeros(num_agent)
            current_ep_num_selected = torch.zeros(num_actions)
            while time_step[env] <= horizon * (i + 1):
                action = select_action(q_values, state, num_selected, time_step, upper_confidence=True)
                reward = env_reward[env, state, action]
                current_ep_num_selected[action] += 1
                next_state = torch.distributions.categorical.Categorical(env_trans_p[env, state, action]).sample()

                #adds to buffer
                buffer[env].extend(format_experience(state, action, reward, next_state))
                state = next_state

                #rollout optimal policy
                optimal_action = all_env_optimal_policy[env, optimal_state]
                optimal_reward = env_reward[env, optimal_state, optimal_action]
                optimal_next_state = torch.distributions.categorical.Categorical(env_trans_p[env, optimal_state, optimal_action]).sample()
                optimal_state = optimal_next_state

                episodic_regret += optimal_reward - reward

                #rollout random policy
                random_action = np.random.randint(0, num_actions, (num_agent,), dtype=int)
                random_reward = env_reward[env, random_state, random_action]
                random_next_state = torch.distributions.categorical.Categorical(
                    env_trans_p[env, random_state, random_action]).sample()
                random_state = random_next_state

                #update reward tracker for debug
                random_track += random_reward
                optimal_track += optimal_reward
                td_track += reward

                time_step[env] += 1

            cumulative_regret[env] += episodic_regret
            num_selected[env] += current_ep_num_selected

        q_values = update_with_experience(q_values, buffer, 1 / (1 + (1 - gamma) * time_step[env]), gamma)
    per_agent_regret = cumulative_regret.mean(-1)
    bayesian_regret = per_agent_regret.mean()
    # print("random: ", random_track.mean())
    # print("optimal: ", optimal_track.mean())
    # print("td: ", td_track.mean())
    # print("qvalues:", q_values)
    return bayesian_regret

def policy_from_mdp(trans_prob, reward, S, A, tol=0.01):
    # performs undiscounted value iteration to output an optimal policy
    value_func = torch.zeros(S)
    policy = torch.zeros(S)
    iter = 0
    gamma = 0.99
    while True:
        diff = torch.zeros(S)
        value = value_func
        action_return = reward + gamma * torch.einsum('ijk,k->ij', trans_prob, value_func)  # [S, A]
        value_func, policy = torch.max(action_return, dim=-1)  # [S]
        diff = torch.maximum(diff, torch.abs(value - value_func))  # [S]

        if torch.any(diff.max() <= tol) or iter >= 10000:
            break
    return policy  # [S]

if __name__ == "__main__":
    seed = 100
    num_states = 20
    num_actions = 10
    num_envs = 10
    num_episodes = 30
    R = 5
    horizon = 75

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    all_env_reward, all_env_trans_p = create_env(num_envs, num_states, num_actions, R, seed)

    # compute optimal policy for each env
    all_env_optimal_policy = torch.zeros((num_envs, num_states), dtype=torch.int64)
    for env in range(num_envs):
        all_env_optimal_policy[env] = policy_from_mdp(
            all_env_trans_p[env], all_env_reward[env], num_states, num_actions, tol=0.001)

    num_agents = [1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for num_agent in num_agents:
        q_values = off_policy_td(num_envs, num_episodes, num_agent, num_states, num_actions, all_env_reward, all_env_trans_p, all_env_optimal_policy, horizon, True)
        print(q_values.item())