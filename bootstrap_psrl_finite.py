import torch
import random
import numpy as np

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

    return policy

class BootstrapPSRL:
    def __init__(self, num_agents, num_envs, S, A, trans_p, rewards, optimal_policy):
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.S = S
        self.A = A
        self.trans_p = trans_p  # [n_envs, n_agents, n_S, n_A, n_S]
        self.optimal_trans_p = trans_p[:, 0, ...]  # [n_envs, n_S, n_A, n_S]
        self.rewards = rewards  # [n_envs, n_agents, n_S, n_A]
        self.optimal_rewards = rewards[:, 0, ...]  # [n_envs, n_S, n_A]
        self.optimal_policy = optimal_policy  # [n_envs, n_S]

        self.alpha = torch.ones(num_envs, S, A, S)
        self.reward_mean = torch.zeros(num_envs, S, A)
        self.reward_scale = torch.ones(num_envs, S, A)

        self.buffer = [[] for _ in range(num_envs)]

    def posterior_sample(self, alpha, mu, scale, n_sample):
        dist_trans_p = torch.distributions.dirichlet.Dirichlet(alpha)
        dist_reward = torch.distributions.normal.Normal(mu, scale)
        trans_p = dist_trans_p.sample([n_sample])
        rewards = dist_reward.sample([n_sample])
        return torch.transpose(trans_p, 0, 1), torch.transpose(rewards, 0, 1)

    def sample(self, num_agents, num_envs, replay_buffer_size):
        indices = torch.randint(low=0, high=replay_buffer_size, size=(num_envs, num_agents, replay_buffer_size))
        sorted, indices = torch.sort(indices)
        return sorted

    # def gather_num_visits(self, sampled_index):
    #     #sampled_index has shape [env, buffer_length]
    #     num_visits = torch.zeros((self.num_envs, self.S, self.A, self.S))
    #     model_reward = torch.zeros((self.num_envs, self.S, self.A))
    #     buffer = torch.tensor(self.buffer)
    #     for env in range(self.num_envs):
    #         for agent in range(self.num_agents):
    #             data_list = buffer[env].gather(0, sampled_index[env, agent].unsqueeze(-1).repeat(1, 4).type(torch.int64)) #gathers the data from buffer based on sampled index
    #             for i in data_list:
    #                 num_visits[env, int(i[0]), int(i[1]), int(i[2])] += 1
    #                 model_reward[env, int(i[0]), int(i[1])] += i[3]
    #     return num_visits, model_reward

    def gather_num_visits(self, sampled_index_from_bootstrap):
        #gathers the num_visits for each environment based on the index sampled from bootstrap
        #sampled_index has shape [env, buffer_length]
        num_visits = torch.zeros((self.num_envs, self.S, self.A, self.S))
        model_reward = torch.zeros((self.num_envs, self.S, self.A))
        buffer = torch.tensor(self.buffer)
        for env in range(self.num_envs):
            data_list = buffer[env].unsqueeze(0).repeat(self.num_agents, 1, 1).gather(1, sampled_index_from_bootstrap[env].unsqueeze(-1).repeat(1, 1, 4).type(torch.int64)) #gathers the data from buffer based on sampled index
            data_list = data_list.reshape(-1, 4)
            for i in data_list:
                num_visits[env, int(i[0]), int(i[1]), int(i[2])] += 1
                model_reward[env, int(i[0]), int(i[1])] += i[3]
        return num_visits, model_reward

    def train(self, episodes, horizon):
        cum_regret = torch.zeros(self.num_envs, self.num_agents)
        num_visits = torch.zeros((self.num_envs, self.S, self.A, self.S))
        model_reward = torch.zeros((self.num_envs, self.S, self.A))

        for i in range(episodes):
            # initialize state tracking for each agent in each env
            curr_states = torch.randint(0, self.S, size=[self.num_envs, self.num_agents])
            # initialize state tracking for the optimal agent in each env
            curr_optimal_state = torch.randint(0, self.S, size=[self.num_envs])

            # (Alg) sample MDP's from the posterior
            # [n_envs, n_agents, n_S, n_A, n_S], [n_envs, n_agents, n_S, n_A]
            sampled_trans_p, sampled_reawrds = self.posterior_sample(
                self.alpha, self.reward_mean, self.reward_scale, self.num_agents)

            # extract optimal policies from sampled MDP's: [n_envs, n_agents, n_S]
            policy = torch.zeros((self.num_envs, self.num_agents, self.S), dtype=torch.int64)
            for env in range(self.num_envs):
                for agent in range(self.num_agents):
                    policy[env, agent, :] = policy_from_mdp(
                        sampled_trans_p[env, agent], sampled_reawrds[env, agent], self.S, self.A)

            for _ in range(horizon):
                # agents rollout
                s_t = curr_states  # [n_envs, n_agents]
                a_t = policy[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1),
                    torch.arange(self.num_agents).unsqueeze(0).unsqueeze(-1),
                    s_t.unsqueeze(-1)].squeeze()  # [n_envs, n_agents]
                trans_p = self.trans_p[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                    torch.arange(self.num_agents).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
                    s_t.unsqueeze(-1).unsqueeze(-1),
                    a_t.unsqueeze(-1).unsqueeze(-1)].squeeze()  # [n_envs, n_agents, n_S]
                # [n_envs, n_agents]
                s_next = torch.distributions.categorical.Categorical(trans_p).sample()
                curr_states = s_next

                # optimal agent rollout
                optimal_s_t = curr_optimal_state
                optimal_a_t = self.optimal_policy[
                    torch.arange(self.num_envs).unsqueeze(-1),
                    optimal_s_t.unsqueeze(-1)].squeeze()  # [n_envs]
                optimal_trans_p = self.optimal_trans_p[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1),
                    optimal_s_t.unsqueeze(-1).unsqueeze(-1),
                    optimal_a_t.unsqueeze(-1).unsqueeze(-1)].squeeze()  # [n_envs, n_S]
                optimal_s_next = torch.distributions.categorical.Categorical(
                    optimal_trans_p).sample()  # [n_envs]
                curr_optimal_state = optimal_s_next

                # collect rewards and update cum_regret
                reward = self.rewards[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                    torch.arange(self.num_agents).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
                    s_t.unsqueeze(-1).unsqueeze(-1),
                    a_t.unsqueeze(-1).unsqueeze(-1)].squeeze()  # [n_envs, n_agents]
                optimal_reward = self.optimal_rewards[
                    torch.arange(self.num_envs).unsqueeze(-1).unsqueeze(-1),
                    optimal_s_t.unsqueeze(-1).unsqueeze(-1),
                    optimal_a_t.unsqueeze(-1).unsqueeze(-1)].squeeze()  # [n_envs]

                cum_regret += optimal_reward.unsqueeze(-1) - reward

                for env in range(self.num_envs):
                    self.buffer[env] += tuple(zip(s_t[env], a_t[env], s_next[env], reward[env]))

            bootstrap_index = self.sample(self.num_agents, self.num_envs, len(self.buffer[0])) #samples each agent's bootstrap index in each environment
            sampled_index = torch.zeros((self.num_envs, self.num_agents, len(self.buffer[0])))
            # samples index used for replay buffer from bootstrap index
            for env in range(self.num_envs):
                indices = torch.randint(low=0, high=len(self.buffer[0]), size=(self.num_agents, len(self.buffer[0])))
                # sampled_index[env] = torch.multinomial(bootstrap_index[env, :, :].type(torch.float32), len(self.buffer[0]), replacement=True)
                sampled_index[env] = torch.gather(bootstrap_index[env], dim=-1, index=indices)
            num_visits, model_reward = self.gather_num_visits(sampled_index)

            # update the posterior of the Dirichlet alpha of transitions
            self.alpha = torch.ones(self.alpha.shape) + num_visits
            # update the posterior of the Gaussian of rewards, [n_envs, n_S, n_A]
            count = torch.ones(self.num_envs, self.S, self.A) + torch.sum(num_visits, dim=-1)
            self.reward_mean = model_reward / count
            self.reward_scale = 1 / torch.sqrt(count)

        # evaluate episodic regret
        per_agent_regret = cum_regret.mean(dim=-1)  # [n_envs]
        per_agent_bayesian_regret = per_agent_regret.mean()
        print("bayesian regret: ", per_agent_bayesian_regret)

        return per_agent_bayesian_regret

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_tensor_type(
            _device_ddtype_tensor_map['cuda'][torch.get_default_dtype()])
        torch.cuda.set_device(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True

    num_states = 20
    num_actions = 10
    num_envs = 10
    num_episodes = 20
    horizon = 10
    # seeds = range(100, 101)
    seeds = (100,)
    for seed in seeds:
        # deterministic settings for current seed
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print("seed: ", seed)

        # define MDP
        all_env_rewards = torch.abs(torch.randn(num_envs, num_states, num_actions))
        dist_trans_p = torch.distributions.dirichlet.Dirichlet(torch.ones(num_states))
        all_env_trans_p = dist_trans_p.sample([num_envs, num_states, num_actions])

        # compute optimal policy for each env
        all_env_optimal_policy = torch.zeros((num_envs, num_states), dtype=torch.int64)
        for env in range(num_envs):
            all_env_optimal_policy[env] = policy_from_mdp(
                all_env_trans_p[env], all_env_rewards[env], num_states, num_actions, tol=0.001)

        regrets = []

        # TODO: there is a bug for num_agents = 1
        list_num_agents = [40, 50, 60, 70] #, 80, 90, 100]
        # list_num_agents = [80, 90, 100]
        for num_agents in list_num_agents:
            print("num of agents: ", num_agents)
            # [n_envs, n_agents, n_S, n_A, n_S]
            all_env_agent_rewards = all_env_rewards.unsqueeze(1).repeat(1, num_agents, 1, 1)
            all_env_agent_trans_p = all_env_trans_p.unsqueeze(1).repeat(1, num_agents, 1, 1, 1)

            bootstrap_psrl = BootstrapPSRL(num_agents, num_envs, num_states, num_actions, all_env_agent_trans_p, all_env_agent_rewards, all_env_optimal_policy)
            regret = bootstrap_psrl.train(num_episodes, horizon)
            regrets.append(regret)

        total_regret = torch.stack(regrets)
        total_regret_np = total_regret.cpu().detach().numpy()

        np.savetxt("results/finite_" + str(seed) + "_agents_" + str(list_num_agents[-1]) + ".csv", np.column_stack((list_num_agents, total_regret_np)), delimiter=",")
