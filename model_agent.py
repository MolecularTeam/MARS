import torch
import torch.nn.functional as F


class ACTOR(torch.nn.Module):
    def __init__(self, n_actions=2, mode='spectral'):
        super().__init__()
        # self.actor = tf.keras.layers.Dense(n_actions, activation=None,
        #                                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=.001, l2=.01))

        self.actor1 = torch.nn.Linear(66, 66*2)  # temporal 64, spectral: 66
        self.bn1 = torch.nn.BatchNorm1d(66*2)

        self.actor4 = torch.nn.Linear(66 * 2, 66)
        self.bn4 = torch.nn.BatchNorm1d(66)

        self.actor5 = torch.nn.Linear(66, n_actions)
        self.bn5 = torch.nn.BatchNorm1d(2)

        self.bias = torch.tensor(0.45)

        self.max_a = torch.tensor(0)
        self.min_a = torch.tensor(1)

    def forward(self, segment):  # B, 64
        segment = self.actor1(segment)
        segment = self.bn1(segment)
        segment = torch.relu(segment)
        segment = self.actor4(segment)
        segment = self.bn4(segment)
        segment = torch.relu(segment)

        segment = self.actor5(segment)

        segment = F.softmax(segment, dim=-1)
        segment[:, 0] = torch.max((segment[:, 0] - self.bias), self.max_a)
        segment[:, 1] = torch.min((segment[:, 1] + self.bias), self.min_a)

        return segment
# Define critic network


class CRITIC(torch.nn.Module):
    def __init__(self, n_actions=2, mode='spectral'):
        super().__init__()
        # self.critic = tf.keras.layers.Dense(1, activation=None,
        #                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=.001, l2=.01))

        self.critic1 = torch.nn.Linear(66 * 2, 66 * 2)  # temporal 64, spectral: 66
        self.bn_1 = torch.nn.BatchNorm1d(66 * 2)

        self.critic4 = torch.nn.Linear(66 * 2, 66)
        self.bn_4 = torch.nn.BatchNorm1d(66)

        self.critic5 = torch.nn.Linear(66, n_actions * n_actions)  # temporal 64, spectral: 66
        self.bn_5 = torch.nn.BatchNorm1d(4)

        self.softsign = torch.nn.Softsign()

    def forward(self, segment):
        segment = self.critic1(segment)
        segment = self.bn_1(segment)
        segment = torch.relu(segment)

        segment = self.critic4(segment)
        segment = self.bn_4(segment)
        segment = torch.relu(segment)

        segment = self.critic5(segment)

        return segment

