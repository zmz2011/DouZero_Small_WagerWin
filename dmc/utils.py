import numpy as np
from collections import Counter
import time
from typing import Dict, List
import logging
import traceback
import torch
from dmc.env_utils import Environment
from env.env import cards2array, Env

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter('[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

Buffer_type = Dict[str, List[torch.Tensor]]


def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = cards2array(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix


def get_batch(free_queue, full_queue, buffers, configs, local_lock):  # already has [role]
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with local_lock:
        indices = [full_queue.get() for _ in range(configs.batch_buffer)]
    batch = {key: buffers[key][indices] for key in buffers}
    for i in indices:
        free_queue.put(i)
    return batch  # no to(device)


def create_optimizers(configs, learner_model):
    """
    Create three optimizers for the three roles
    """
    roles = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for r in roles:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(r),
            lr=configs.learning_rate,
            momentum=configs.momentum,
            eps=configs.eps,
            alpha=configs.alpha)
        optimizers[r] = optimizer
    return optimizers


def create_buffers_new(configs):
    """
    new buffers for new model training
    """
    T = configs.buffer_size
    N = configs.num_buffers
    device = configs.training_device  # the fixed device
    roles = ['landlord', 'landlord_up', 'landlord_down']
    buffers = {}  # {role: {data: [Tensor]*num_buffers}
    specs = {
        'done': {'size': (), 'dtype': torch.bool},
        'episode_return': {'size': (), 'dtype': torch.float32},
        'obs_x': {'size': (-1,), 'dtype': torch.float32},
        'obs_z': {'size': (4, 114), 'dtype': torch.float32},
        # new targets
        'target_u': {'size': (), 'dtype': torch.float32},
        'target_R': {'size': (), 'dtype': torch.float32},  # for both v_w+v_l
    }

    for r in roles:
        buffers[r] = {}
        x_dim = 263 if r == 'landlord' else 341  #
        specs['obs_x']['size'] = (x_dim,)
        for key in specs:
            buffers[r][key] = torch.zeros((N, T, *specs[key]['size']), requires_grad=False,
                                             dtype=specs[key]['dtype']).to(device).share_memory_()
    return buffers


def act_proc_new(i, device, free_queue, full_queue, actor_models, buffers, configs, epsilon):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.

    configs.model_type = 0/1
    """
    roles = ['landlord', 'landlord_up', 'landlord_down']
    try:
        T = configs.buffer_size
        log.info('Device %s Actor %i started.', str(device), i)

        env = Environment(Env(configs.objective), device)

        # all in tensor cpu
        done_buf = {p: [] for p in roles}
        reward_buf = {p: [] for p in roles}  # not for training
        # new targets
        u_buf = {p: [] for p in roles}  # 0/1
        R_buf = {p: [] for p in roles}  # true payoff

        obs_z_buf = {p: [] for p in roles}
        obs_x_buf = {p: [] for p in roles}  # obs_x only
        size = {p: 0 for p in roles}

        role, obs, env_outputs = env.initial()

        while True:
            while True:
                z = obs['z'].to(device)
                x_batch = obs['x_batch'].to(device)
                if len(obs['legal_actions']) > 1:
                    with torch.no_grad():
                        if configs.model_type == 0:  # Original
                            idx = actor_models.models[role].actor_forward(z, x_batch, epsilon=epsilon.value)
                        else:  # New Model
                            bomb_val = pow(2., env_outputs['bomb_played'])
                            idx = actor_models.models[role].actor_forward(z, x_batch, epsilon=epsilon.value,
                                                                          Q_w_min=bomb_val, Q_l_max=-bomb_val)
                        _idx = int(idx.cpu().detach().item())
                else:
                    _idx = 0
                action = obs['legal_actions'][_idx]
                obs_z_buf[role].append(obs['z'])
                obs_x_buf[role].append(obs['x_batch'][_idx])
                size[role] += 1

                role, obs, env_outputs = env.step(action)

                if env_outputs['done']:
                    for r in roles:
                        diff = size[r] - len(R_buf[r])
                        if diff > 0:
                            done_buf[r].extend([False for _ in range(diff - 1)])
                            done_buf[r].append(True)
                            if r == 'landlord':
                                episode_return = env_outputs['episode_return']
                            else:
                                episode_return = -env_outputs['episode_return']

                            reward_buf[r].extend([0. for _ in range(diff - 1)])
                            reward_buf[r].append(episode_return)
                            R_buf[r].extend([episode_return for _ in range(diff)])
                            if episode_return > 0:  # win
                                u_buf[r].extend([1. for _ in range(diff)])
                            else:  # lose
                                u_buf[r].extend([0. for _ in range(diff)])
                    break

            for r in roles:
                while size[r] > T:
                    index = free_queue[r].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[r]['done'][index][t] = done_buf[r][t]
                        buffers[r]['episode_return'][index][t] = reward_buf[r][t]
                        if configs.model_type == 1:  # new model
                            buffers[r]['target_u'][index][t] = u_buf[r][t]
                        buffers[r]['target_R'][index][t] = R_buf[r][t]
                        buffers[r]['obs_x'][index][t] = obs_x_buf[r][t]
                        buffers[r]['obs_z'][index][t] = obs_z_buf[r][t]
                    full_queue[r].put(index)

                    done_buf[r] = done_buf[r][T:]
                    reward_buf[r] = reward_buf[r][T:]
                    R_buf[r] = R_buf[r][T:]
                    u_buf[r] = u_buf[r][T:]
                    obs_z_buf[r] = obs_z_buf[r][T:]
                    obs_x_buf[r] = obs_x_buf[r][T:]
                    size[r] -= T

    except KeyboardInterrupt:
        info = "actor" + str(i) + ": KeyboardInterrupt"
        # info += ", (epsilon: " + str(epsilon.value)
        log.info(info)
        return

    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        raise e
