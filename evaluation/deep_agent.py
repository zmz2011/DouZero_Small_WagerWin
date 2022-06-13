from env.env import get_obs_new
from dmc.models import *


class LandlordLstmAgent:
    def __init__(self, model_path=None):
        self.model = LandlordLstmModel()
        if model_path is not None:
            pretrained = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(pretrained)
        self.model.eval()
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs_new(infoset)
        z = torch.from_numpy(obs['z']).to(self.device)
        x_batch = torch.from_numpy(obs['x_batch']).to(self.device)
        with torch.no_grad():
            idx_a = self.model.actor_forward(z, x_batch, 0.)
        idx_a = int(idx_a.detach().cpu().item())
        best_action = infoset.legal_actions[idx_a]
        return best_action


class PeasantLstmAgent:
    def __init__(self, model_path=None):
        self.model = PeasantLstmModel()
        if model_path is not None:
            pretrained = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(pretrained)
        self.model.eval()
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        if infoset.player_role == 'landlord_down' or infoset.player_role == 'landlord_up':
            obs = get_obs_new(infoset)
        else:
            raise ValueError("PeasantLstmAgent act(): Error player_role")

        z = torch.from_numpy(obs['z']).to(self.device)
        x_batch = torch.from_numpy(obs['x_batch']).to(self.device)
        with torch.no_grad():
            idx_a = self.model.actor_forward(z, x_batch, 0.)
        idx_a = int(idx_a.detach().cpu().item())
        best_action = infoset.legal_actions[idx_a]
        return best_action


class NewLandlordLstmAgent:
    def __init__(self, model_path=None):
        self.model = NewLandlordLstmModel()
        if model_path is not None:
            pretrained = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(pretrained)
        self.model.eval()
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs_new(infoset)

        z = torch.from_numpy(obs['z']).to(self.device)
        x_batch = torch.from_numpy(obs['x_batch']).to(self.device)
        bomb_val = pow(2, infoset.bomb_played)
        with torch.no_grad():
            idx_a = self.model.actor_forward(z, x_batch, epsilon=0., Q_w_min=bomb_val, Q_l_max=-bomb_val)
        idx_a = int(idx_a.detach().cpu().item())
        best_action = infoset.legal_actions[idx_a]
        return best_action


class NewPeasantLstmAgent:
    def __init__(self, model_path=None):
        self.model = NewPeasantLstmModel()
        if model_path is not None:
            pretrained = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(pretrained)
        self.model.eval()
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        if infoset.player_role == 'landlord_down' or infoset.player_role == 'landlord_up':
            obs = get_obs_new(infoset)
        else:
            raise ValueError("PeasantLstmAgent act(): Error player_role")

        z = torch.from_numpy(obs['z']).to(self.device)
        x_batch = torch.from_numpy(obs['x_batch']).to(self.device)
        bomb_val = pow(2, infoset.bomb_played)
        with torch.no_grad():
            idx_a = self.model.actor_forward(z, x_batch, epsilon=0., Q_w_min=bomb_val, Q_l_max=-bomb_val)
        idx_a = int(idx_a.detach().cpu().item())
        best_action = infoset.legal_actions[idx_a]
        return best_action
