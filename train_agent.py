import torch
import torch.nn.functional as F


def cross_prod(pi_a1, pi_a2, N_action=2):
    new_pi = torch.zeros(pi_a1.shape[0], N_action * N_action, device=pi_a2.device)
    for i in range(N_action):
        for j in range(N_action):
            new_pi[:, i * N_action + j] = pi_a1[:, i] * pi_a2[:, j]
    return new_pi


def mars_agent_gradient_oneshot(model, actor1, actor2, critic, inputs, feature, labels, state1_list, state_next1_list,
                                state2_list, state_next2_list, pi_a1_list_list, pi_a2_list_list, action_probs_temp_list, action_probs_spec_list,
                                a1_list_list, a2_list_list, optc, opta1, opta2, criterion, epoch, gam=0.96, r_gam=10):
    gamma = gam  # discount factor
    r_gamma = r_gam
    N_action = 2

    with torch.no_grad():
        loss_FM = criterion(model(inputs), labels)
        loss_AM = criterion(model.classifier(feature), labels)
    # Reward, r_t

    reward = loss_FM - loss_AM

    num_step = len(state1_list)

    critic_loss_list = torch.tensor([0.0], device=inputs.device)
    a1_loss_list = torch.tensor([0.0], device=inputs.device)
    a2_loss_list = torch.tensor([0.0], device=inputs.device)
    Qmean_list = torch.tensor([0.0], device=inputs.device)

    for t in range(num_step):
        a1_list, a2_list = a1_list_list[t], a2_list_list[t]
        state1, state2 = state1_list[t], state2_list[t]
        state_next1, state_next2 = state_next1_list[t], state_next2_list[t]
        action_probs_temp, action_probs_spec = action_probs_temp_list[t], action_probs_spec_list[t]
        pi_a1_list, pi_a2_list = pi_a1_list_list[t], pi_a2_list_list[t]

        a1_list = a1_list[:, 0]
        a2_list = a2_list[:, 0]

        state = torch.cat([state1, state2], dim=-1)
        state_next = torch.cat([state_next1, state_next2], dim=-1)
        Q = critic(state)
        Q_est = Q.clone()
        Qmean_list += Q.clone().detach().mean()
        Q_prime = critic(state_next)
        Q_est_prime = Q_prime.clone()
        a_index = a1_list * N_action + a2_list
        Q_est[:, a_index] = reward * r_gamma + (gamma * torch.sum(cross_prod(action_probs_temp, action_probs_spec) * Q_est_prime, dim=-1))

        critic_loss = 1.0 * (Q.clone() - Q_est.detach()).square().mean()
        critic_loss_list += critic_loss
        temp_Q1 = torch.zeros([Q.shape[0], N_action], device=Q.device)
        # for a1 in range(N_action):
        #     temp_Q1[:, a1] = Q[:, a1 * N_action + a2_list]
        a_index = (a1_list * N_action + a2_list).clone().detach()
        temp_A1 = torch.gather(Q, -1, a_index.unsqueeze(-1)).squeeze() - torch.sum(pi_a1_list * temp_Q1, dim=-1)

        a1_loss = (-temp_A1 * torch.gather(pi_a1_list, -1, a1_list.unsqueeze(-1)).squeeze().log()).mean()
        a1_loss_list += a1_loss

        temp_Q2 = torch.zeros([Q.shape[0], N_action], device=Q.device)
        # for a2 in range(N_action):
        #     temp_Q2[:, a2] = Q[:, a1_list * N_action + a2]
        a_index = (a1_list * N_action + a2_list).clone().detach()
        temp_A2 = torch.gather(Q, -1, a_index.unsqueeze(-1)).squeeze() - torch.sum(pi_a2_list * temp_Q2, dim=-1)

        a2_loss = (-temp_A2 * torch.gather(pi_a2_list, -1, a2_list.unsqueeze(-1)).squeeze().log()).mean()
        a2_loss_list += a2_loss

    # wandb.log({f"reward": reward.mean(),
    #            f"meanQ": Qmean_list / num_step},
    #           step=epoch
    #           )

    return critic_loss_list / num_step, a1_loss_list / num_step, a2_loss_list / num_step


def agent_gradient(model, actor, critic, inputs, feature, labels, state, state_next, criterion):
    gamma = 0.95  # discount factor
    N_action = 2
    loss_FM = criterion(model(inputs), labels)

    loss_AM = criterion(model.classifier(feature), labels)

    # Reward, r_t
    reward = loss_FM - loss_AM

    # print("REWARD", reward.shape)
    advantage = reward.unsqueeze(-1) + gamma * critic(state_next) - critic(state)
    # print("advantage", advantage.shape)
    critic_loss = 0.5 * advantage.square().mean()

    actor_loss = (-F.log_softmax(actor(state),
                                 dim=-1) * advantage).mean()
    # print("actor_loss", actor_loss.shape)
    return critic_loss, actor_loss
