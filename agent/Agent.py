import torch

from utils import down_scale_trj


class Agent:
    def __init__(self, policy, environment, horizon):
        self.policy = policy
        self.environment = environment
        self.horizon = horizon

    def sample_trajectory(self, number_trajectories, print_traj=False):

        with torch.no_grad():
            trajectory = []

            disturbances_list = []
            states_list = []
            actions_list = []
            one_hot_actions_list = []
            reward_list = []

            self.environment.reset(number_trajectories)

            for t in range(self.horizon):
                # current state(s) in the systems
                state = self.environment.state

                # actions
                action_logits = self.policy.forward(self.environment.state)
                one_hot_action = self.policy.distribution(action_logits).sample()
                # next step
                _, disturbance, reward, action_value = self.environment.step(one_hot_action)

                if print_traj:
                    print('state : ', state)
                    print('Proba actions : ', torch.nn.functional.softmax(action_logits, dim=1))
                    print('actions : ', action_value, one_hot_action)
                    print('disturbance : ', disturbance)
                    print('reward : ', reward)

                disturbances_list.append(disturbance)
                actions_list.append(action_value)
                one_hot_actions_list.append(one_hot_action)
                reward_list.append(reward)
                states_list.append(state)
                trajectory.append((state.squeeze().tolist(),
                                   action_value.squeeze().tolist(),
                                   disturbance.squeeze().tolist(),
                                   reward.squeeze().tolist()))

            return (trajectory,
                    torch.stack(states_list, dim=1).squeeze(dim=0),
                    torch.stack(disturbances_list, dim=1).squeeze(dim=0),
                    torch.stack(actions_list, dim=1).squeeze(dim=0),
                    torch.stack(one_hot_actions_list, dim=1).squeeze(dim=0),
                    torch.stack(reward_list, dim=1).squeeze(dim=0))

    def avg_performance(self, mc_samples):
        _, states, _, _, _, rew = self.sample_trajectory(mc_samples)

        avg_rew = torch.mean(torch.sum(rew, dim=1), dim=0).squeeze().tolist()
        sys_deviation = self.environment.system_nn.stabilization_estimate(states)

        return avg_rew, sys_deviation

    def evaluate_performance(self, number_trajectories):
        _, states, dist, actions, _, rewards = self.sample_trajectory(number_trajectories)
        states, actions, rewards = down_scale_trj(self.environment.system_nn, states, actions, rewards)
        self.environment.system_nn.unwrapped.render(states, actions, dist, rewards, number_trajectories)
