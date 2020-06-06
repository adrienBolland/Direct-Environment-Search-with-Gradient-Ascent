import torch


def system_error(system_nn, pol_nn, states_batch, dist_batch, oha_batch, rew_batch):
    with torch.no_grad():
        states_batch = states_batch
        dist_batch = dist_batch
        oha_batch = oha_batch
        rew_batch = rew_batch
        mean_rew = torch.mean(torch.sum(rew_batch, dim=1)).squeeze().item()

    # compute the states with gradient
    horizon = states_batch.shape[1]

    states_batch_grad = torch.zeros(states_batch.shape, device=system_nn.unwrapped.device)
    states_batch_grad[:, 0, :] = states_batch[:, 0, :]

    for t in range(horizon - 1):
        states_batch_grad[:, t + 1, :] = system_nn.dynamics(states_batch_grad[:, t, :].clone(),
                                                            oha_batch[:, t, :],
                                                            dist_batch[:, t, :])

    # compute the error
    batch_size = states_batch.shape[0]
    loss_batch = []

    for batch in range(batch_size):
        err = _system_loss(system_nn, pol_nn, states_batch_grad[batch, :, :],
                           dist_batch[batch, :, :], oha_batch[batch, :, :], rew_batch[batch, :, :], mean_rew)
        loss_batch.append(err)

    return torch.mean(torch.stack(loss_batch))


def _system_loss(system_nn, pol_nn, states, disturbances, one_hot_actions, rew, baseline=0.):
    logits_grad = pol_nn.forward(states)
    sum_log_pol_grad = torch.sum(pol_nn.distribution(logits_grad).log_prob(one_hot_actions))
    sum_log_dist_grad = torch.sum(system_nn.disturbance(states, one_hot_actions).log_prob(disturbances))
    sum_log_p_grad = sum_log_pol_grad + sum_log_dist_grad
    rew_grad = system_nn.reward(states, one_hot_actions, disturbances)

    return -(sum_log_p_grad * (torch.sum(rew) - baseline) + torch.sum(rew_grad))
