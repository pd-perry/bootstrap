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

def CreateEnv(num_env, num_states, num_actions, R, seed):
    reward = torch.abs(torch.randn(num_env, num_states, num_actions))
    dist_trans_p = torch.distributions.dirichlet.Dirichlet(torch.ones(num_states))
    trans_p = dist_trans_p.sample([num_env, num_states, num_actions])
    return reward, trans_p

def Sarsa(num_env, num_episodes, num_agent, num_states, num_actions, env_reward, env_trans_p, all_env_optimal_policy, horizon, upper_confidence=False):
    gamma = 0.9
    learning_rate = 0.1
    epsilon_greedy = 0.2
    num_selected = torch.ones((num_env, num_actions, num_agent))
    q_values = torch.rand((num_env, num_states, num_actions, num_agent))
    cumulative_regret = torch.zeros((num_env, num_agent))

    for i in range(num_episodes):
        for env in range(num_env):
            state = torch.randint(0, num_states, (num_agent,))
            action = torch.argmax(q_values[env, state, :, torch.arange(num_agent)], -1)

            optimal_state = state
            optimal_action = all_env_optimal_policy[env, optimal_state]

            time_step = 1
            episodic_regret = torch.zeros(num_agent)
            while time_step <= horizon:
                reward = env_reward[env, state, action]
                num_selected[env, action] += 1
                next_state = torch.distributions.categorical.Categorical(env_trans_p[env, state, action]).sample()

                if upper_confidence:
                    upper_confidence_q = q_values[env].clone() + \
                                         torch.sqrt(torch.div(torch.full_like(num_selected[env], np.log(time_step)),
                                                              num_selected[env])).unsqueeze(0).repeat(num_states, 1, 1)
                    next_action = torch.argmax(upper_confidence_q[next_state, :, torch.arange(num_agent)], -1)
                else:
                    explore = np.random.random(1) < epsilon_greedy
                    if explore:
                        next_action = np.random.randint(0, num_actions, 1)
                    else:
                        next_action = np.argmax(q_values[next_state])

                q_values[env, state, action, torch.arange(num_agent)] += learning_rate * \
                                        (reward + gamma * q_values[env, next_state, next_action, torch.arange(num_agent)] - \
                                         q_values[env, state, action, torch.arange(num_agent)])
                state = next_state
                action = next_action

                #rollout optimal policy
                optimal_reward = env_reward[env, optimal_state, optimal_action]
                optimal_next_state = torch.distributions.categorical.Categorical(env_trans_p[env, optimal_state, optimal_action]).sample()
                optimal_next_action = all_env_optimal_policy[env, optimal_next_state]
                optimal_state = optimal_next_state
                optimal_action = optimal_next_action

                episodic_regret += optimal_reward - reward

                time_step += 1
            cumulative_regret[env] += episodic_regret

    per_agent_regret = cumulative_regret.mean(-1)
    bayesian_regret = per_agent_regret.mean()
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

    all_env_reward, all_env_trans_p = CreateEnv(num_envs, num_states, num_actions, R, seed)

    # compute optimal policy for each env
    all_env_optimal_policy = torch.zeros((num_envs, num_states), dtype=torch.int64)
    for env in range(num_envs):
        all_env_optimal_policy[env] = policy_from_mdp(
            all_env_trans_p[env], all_env_reward[env], num_states, num_actions, tol=0.001)

    #TODO: OPTIMAL REWARD IS 4450 as opposed to 4150
    num_agents = [2, 4, 6, 8, 10]

    for num_agent in num_agents:
        q_values = Sarsa(num_envs, num_episodes, num_agent, num_states, num_actions, all_env_reward, all_env_trans_p, all_env_optimal_policy, horizon, True)
        print(q_values)