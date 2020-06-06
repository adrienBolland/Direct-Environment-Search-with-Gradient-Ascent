from torch.optim import Adam

from agent.Agent import Agent
from environment.Environment import Environment
from errors.errors_sys_search import system_error


def optimize(system_nn, pol_nn, log_writer, **kwargs):
    return optimize_joint(system_nn, pol_nn, log_writer, **kwargs)


def optimize_joint(system_nn, pol_nn, log_writer, **kwargs):
    # unpack kwargs
    horizon = kwargs.get("horizon")
    nb_iterations = kwargs.get("nb_iterations")
    batch_size = kwargs.get("batch_size")

    policy_fit = kwargs.get("policy", False)
    system_fit = kwargs.get("system", False)

    mc_samples = kwargs.get("mc_samples", 128)

    env = Environment(system_nn)
    agent = Agent(pol_nn, env, horizon)

    # Optimizers
    parameters_list = []
    if policy_fit:
        parameters_list = parameters_list + list(pol_nn.parameters())

    if system_fit:
        parameters_list = parameters_list + list(system_nn.parameters())
    if parameters_list:
        lr = kwargs.get("learning_rate", .001)
        optimizer = Adam(parameters_list, lr=lr)

        for it in range(nb_iterations):
            loss = {}
            params = {}

            # set gradient to zero
            optimizer.zero_grad()

            # generate the batch
            _, states_batch, dist_batch, _, oha_batch, rew_batch = agent.sample_trajectory(batch_size)

            # Loss #
            system_loss = system_error(system_nn, pol_nn, states_batch, dist_batch, oha_batch, rew_batch)

            system_loss.backward(retain_graph=policy_fit)

            optimizer.step()
            system_nn.project_parameters()
            pol_nn.project_parameters()

            if system_fit and log_writer is not None:
                params['system'] = system_nn.unwrapped.named_parameters()
                log_writer.add_system_parameters(system_nn.parameters_dict(), step=it)

            if policy_fit and log_writer is not None:
                params['policy'] = pol_nn.named_parameters()
                actions = pol_nn(states_batch)  # (B, H, A), need to stack along the B dim
                log_writer.add_policy_histograms(actions.view(-1, actions.shape[2]), step=it)

            if log_writer is not None:
                loss['loss'] = system_loss.item()

                log_writer.add_grad_histograms(params, step=it)
                log_writer.add_loss(loss, step=it)

                # performance of the agent on the epoch
                ep_perf, return_estimate = agent.avg_performance(mc_samples)
                log_writer.add_expected_return(ep_perf, step=it)
                log_writer.add_return(return_estimate, step=it)

    return env, agent
