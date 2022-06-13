"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np
import torch
from torch import nn


class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(114, 128, batch_first=True)  #
        self.dense1 = nn.Linear(263 + 128, 512)  #
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def actor_forward(self, z, x_batch, epsilon=0.):  # accelerate actor speed by skipping z_batch
        lstm_out, _ = self.lstm(z.unsqueeze(0))
        lstm_batch = lstm_out[0][-1].repeat(x_batch.shape[0], 1)
        x_batch = torch.cat([lstm_batch, x_batch], dim=-1)
        hid = self.dense1(x_batch)
        hid = torch.relu(hid)
        hid = self.dense2(hid)
        hid = torch.relu(hid)
        hid = self.dense3(hid)
        hid = torch.relu(hid)
        hid = self.dense4(hid)
        hid = torch.relu(hid)
        hid = self.dense5(hid)
        hid = torch.relu(hid)
        Q = self.dense6(hid).squeeze(-1)
        if epsilon > 0. and np.random.rand() < epsilon:
            action = torch.randint(x_batch.shape[0], ())
        else:
            action = torch.argmax(Q, dim=0)
        return action

    def training_forward(self, z_batch, x_batch):  # for training
        lstm_out, _ = self.lstm(z_batch)
        lstm_out = lstm_out[:, -1, :]
        x_batch = torch.cat([lstm_out, x_batch], dim=-1)
        hid = self.dense1(x_batch)
        hid = torch.relu(hid)
        hid = self.dense2(hid)
        hid = torch.relu(hid)
        hid = self.dense3(hid)
        hid = torch.relu(hid)
        hid = self.dense4(hid)
        hid = torch.relu(hid)
        hid = self.dense5(hid)
        hid = torch.relu(hid)
        Q = self.dense6(hid).squeeze(-1)
        return Q

    def debug(self, z, x_batch):  # for debugging & testing
        lstm_out, _ = self.lstm(z.unsqueeze(0))
        lstm_batch = lstm_out[0][-1].repeat(x_batch.shape[0], 1)
        x_batch = torch.cat([lstm_batch, x_batch], dim=-1)
        hid = self.dense1(x_batch)
        hid = torch.relu(hid)
        hid = self.dense2(hid)
        hid = torch.relu(hid)
        hid = self.dense3(hid)
        hid = torch.relu(hid)
        hid = self.dense4(hid)
        hid = torch.relu(hid)
        hid = self.dense5(hid)
        hid = torch.relu(hid)
        Q = self.dense6(hid).squeeze(-1)
        action = torch.argmax(Q, dim=0)
        return (Q,), action


class PeasantLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(114, 128, batch_first=True)
        self.dense1 = nn.Linear(341 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def actor_forward(self, z, x_batch, epsilon=0.):  # accelerate actor speed by skipping z_batch
        lstm_out, _ = self.lstm(z.unsqueeze(0))
        lstm_batch = lstm_out[0][-1].repeat(x_batch.shape[0], 1)
        x_batch = torch.cat([lstm_batch, x_batch], dim=-1)
        hid = self.dense1(x_batch)
        hid = torch.relu(hid)
        hid = self.dense2(hid)
        hid = torch.relu(hid)
        hid = self.dense3(hid)
        hid = torch.relu(hid)
        hid = self.dense4(hid)
        hid = torch.relu(hid)
        hid = self.dense5(hid)
        hid = torch.relu(hid)
        Q = self.dense6(hid).squeeze(-1)
        if epsilon > 0. and np.random.rand() < epsilon:
            action = torch.randint(x_batch.shape[0], ())
        else:
            action = torch.argmax(Q, dim=0)
        return action

    def training_forward(self, z_batch, x_batch):  # for training
        lstm_out, _ = self.lstm(z_batch)
        lstm_out = lstm_out[:, -1, :]
        x_batch = torch.cat([lstm_out, x_batch], dim=-1)
        hid = self.dense1(x_batch)
        hid = torch.relu(hid)
        hid = self.dense2(hid)
        hid = torch.relu(hid)
        hid = self.dense3(hid)
        hid = torch.relu(hid)
        hid = self.dense4(hid)
        hid = torch.relu(hid)
        hid = self.dense5(hid)
        hid = torch.relu(hid)
        Q = self.dense6(hid).squeeze(-1)
        return Q

    def debug(self, z, x_batch):
        lstm_out, _ = self.lstm(z.unsqueeze(0))
        lstm_batch = lstm_out[0][-1].repeat(x_batch.shape[0], 1)
        x_batch = torch.cat([lstm_batch, x_batch], dim=-1)
        hid = self.dense1(x_batch)
        hid = torch.relu(hid)
        hid = self.dense2(hid)
        hid = torch.relu(hid)
        hid = self.dense3(hid)
        hid = torch.relu(hid)
        hid = self.dense4(hid)
        hid = torch.relu(hid)
        hid = self.dense5(hid)
        hid = torch.relu(hid)
        Q = self.dense6(hid).squeeze(-1)
        action = torch.argmax(Q, dim=0)
        return (Q,), action


class Models:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """

    def __init__(self, device):
        self.models = {
            'landlord': LandlordLstmModel().to(device),
            'landlord_up': PeasantLstmModel().to(device),
            'landlord_down': PeasantLstmModel().to(device)
        }

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()

    def train(self):
        self.models['landlord'].train()
        self.models['landlord_up'].train()
        self.models['landlord_down'].train()

    def parameters(self, role):
        return self.models[role].parameters()


class NewLandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(114, 128, batch_first=True)
        self.mlps = nn.Sequential(
            nn.Linear(128 + 263, 512),  #
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),  # change to 3
        )

    def actor_forward(self, z, x_batch, epsilon=0., Q_w_min=1., Q_l_max=-1.):  # acceleration in actor sampling
        lstm_out, _ = self.lstm(z.unsqueeze(0))
        lstm_batch = lstm_out[0][-1].repeat(x_batch.shape[0], 1)
        x_batch = torch.cat([lstm_batch, x_batch], dim=-1)
        out = self.mlps(x_batch)

        pr, Q_w, Q_l = torch.split(out, (1, 1, 1), dim=-1)
        pr = torch.sigmoid(pr).squeeze(-1)  # win/lose prob (0, 1)
        Q_w = torch.clamp(Q_w.squeeze(-1), Q_w_min, 64.)  # adp when win
        Q_l = torch.clamp(Q_l.squeeze(-1), -64., Q_l_max)  # adp when lose
        Q = pr * Q_w + (1. - pr) * Q_l  # real payoff
        if epsilon > 0. and np.random.rand() < epsilon:
            action = torch.randint(Q.shape[0], ())
        else:
            action = torch.argmax(Q, dim=0)
        return action

    def training_forward(self, z_batch, x_batch):
        lstm_out, _ = self.lstm(z_batch)
        lstm_out = lstm_out[:, -1, :]
        x_batch = torch.cat([lstm_out, x_batch], dim=-1)
        out = self.mlps(x_batch)
        pr, Q_w, Q_l = torch.split(out, (1, 1, 1), dim=-1)
        pr = torch.sigmoid(pr).squeeze(-1)
        return pr, Q_w.squeeze(-1), Q_l.squeeze(-1)

    def debug(self, z, x_batch, Q_w_min=1., Q_l_max=-1.,
              weights=None):  # weights:(a^w, b^w, a^l, b^l) (default:(1,0,1,0))
        lstm_out, _ = self.lstm(z.unsqueeze(0))
        lstm_batch = lstm_out[0][-1].repeat(x_batch.shape[0], 1)
        x_batch = torch.cat([lstm_batch, x_batch], dim=-1)
        out = self.mlps(x_batch)

        pr, Q_w, Q_l = torch.split(out, (1, 1, 1), dim=-1)
        pr = torch.sigmoid(pr).squeeze(-1)  # win/lose prob (0, 1)
        Q_w_c = torch.clamp(Q_w.squeeze(-1), Q_w_min, 64.)  # adp when win
        Q_l_c = torch.clamp(Q_l.squeeze(-1), -64., Q_l_max)  # adp when lose
        if weights is not None:
            Q = pr * (weights[0] * Q_w_c + weights[1]) + \
                  (1. - pr) * (weights[2] * Q_l_c + weights[3])  # customized policy adaptation
        else:
            Q = pr * Q_w_c + (1. - pr) * Q_l_c  # real payoff
        action = torch.argmax(Q, dim=0)
        return (pr, Q_w.squeeze(-1), Q_l.squeeze(-1), Q, out), action


class NewPeasantLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(114, 128, batch_first=True)
        self.mlps = nn.Sequential(
            nn.Linear(128 + 341, 512),  #
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),  # change to 3
        )

    def actor_forward(self, z, x_batch, epsilon=0., Q_w_min=1., Q_l_max=-1.):  # acceleration in actor sampling
        lstm_out, _ = self.lstm(z.unsqueeze(0))
        lstm_batch = lstm_out[0][-1].repeat(x_batch.shape[0], 1)
        x_batch = torch.cat([lstm_batch, x_batch], dim=-1)
        out = self.mlps(x_batch)

        pr, Q_w, Q_l = torch.split(out, (1, 1, 1), dim=-1)
        pr = torch.sigmoid(pr).squeeze(-1)  # win/lose prob (0, 1)
        Q_w = torch.clamp(Q_w.squeeze(-1), Q_w_min, 64.)  # adp when win
        Q_l = torch.clamp(Q_l.squeeze(-1), -64., Q_l_max)  # adp when lose
        Q = pr * Q_w + (1. - pr) * Q_l  # real payoff
        if epsilon > 0. and np.random.rand() < epsilon:
            action = torch.randint(Q.shape[0], ())
        else:
            action = torch.argmax(Q, dim=0)
        return action

    def training_forward(self, z_batch, x_batch):
        lstm_out, _ = self.lstm(z_batch)
        lstm_out = lstm_out[:, -1, :]
        x_batch = torch.cat([lstm_out, x_batch], dim=-1)
        out = self.mlps(x_batch)
        pr, Q_w, Q_l = torch.split(out, (1, 1, 1), dim=-1)
        pr = torch.sigmoid(pr).squeeze(-1)
        return pr, Q_w.squeeze(-1), Q_l.squeeze(-1)

    def debug(self, z, x_batch, Q_w_min=1., Q_l_max=-1.,
              weights=None):  # weights:(a^w, b^w, a^l, b^l) (default:(1,0,1,0))
        lstm_out, _ = self.lstm(z.unsqueeze(0))
        lstm_batch = lstm_out[0][-1].repeat(x_batch.shape[0], 1)
        x_batch = torch.cat([lstm_batch, x_batch], dim=-1)
        out = self.mlps(x_batch)

        pr, Q_w, Q_l = torch.split(out, (1, 1, 1), dim=-1)
        pr = torch.sigmoid(pr).squeeze(-1)  # win/lose prob (0, 1)
        Q_w_c = torch.clamp(Q_w.squeeze(-1), Q_w_min, 64.)  # adp when win
        Q_l_c = torch.clamp(Q_l.squeeze(-1), -64., Q_l_max)  # adp when lose
        if weights is not None:
            Q = pr * (weights[0] * Q_w_c + weights[1]) + \
                  (1. - pr) * (weights[2] * Q_l_c + weights[3])  # customized policy adaptation
        else:
            Q = pr * Q_w_c + (1. - pr) * Q_l_c  # real payoff
        action = torch.argmax(Q, dim=0)
        return (pr, Q_w.squeeze(-1), Q_l.squeeze(-1), Q, out), action


class NewModels:
    def __init__(self, device):
        self.models = {
            'landlord': NewLandlordLstmModel().to(device),
            'landlord_up': NewPeasantLstmModel().to(device),
            'landlord_down': NewPeasantLstmModel().to(device)
        }

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()

    def train(self):
        self.models['landlord'].train()
        self.models['landlord_up'].train()
        self.models['landlord_down'].train()

    def parameters(self, role):
        return self.models[role].parameters()
