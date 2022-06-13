"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of manually resetting
the environment, we do it automatically.
"""
import numpy as np
import torch


def _format_observation(obs):
    position = obs['position']
    x_batch = torch.from_numpy(obs['x_batch'])
    x_no_action = torch.from_numpy(obs['x_no_action'])
    z = torch.from_numpy(obs['z'])  # no z_batch
    obs = {'x_batch': x_batch, 'legal_actions': obs['legal_actions'], 'z': z}
    return position, obs, x_no_action


class Environment:
    def __init__(self, env, device):
        """
        Initialize this environment wrapper
        """
        self.env = env
        # self.device = device
        self.episode_return = None

    def initial(self, card_play_data=None):
        position, obs, x_no_action = _format_observation(self.env.reset(card_play_data))
        self.episode_return = 0.

        return position, obs, dict(
            done=False,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            bomb_played=0  # new item
            # obs_z=z,
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.episode_return += reward
        episode_return = self.episode_return

        if done:  # automatic reset env after done
            obs = self.env.reset()
            self.episode_return = 0.

        position, obs, x_no_action = _format_observation(obs)

        return position, obs, dict(
            done=done,  # bool
            episode_return=episode_return,  # float (payoff of landlord)
            obs_x_no_action=x_no_action,
            bomb_played=info['bomb_played']
            # obs_z=z,
        )

    def close(self):
        self.env.close()
