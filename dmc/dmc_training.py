import os
import threading
import time
import timeit
import pprint
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn

from dmc.models import NewModels, Models
from dmc.utils import log, get_batch, create_optimizers, create_buffers_new, act_proc_new

role_list = ['landlord', 'landlord_up', 'landlord_down']


def update_loss_new(role: str, actor_models_dict, training_model, batch, optim, configs, learn_lock, max_grad_norm):
    """separated pr/Q_w/Q_l loss for new models"""
    obs_x = torch.flatten(batch['obs_x'].to(configs.training_device), 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(configs.training_device), 0, 1)

    target_R = torch.flatten(batch['target_R'].to(configs.training_device), 0, 1)
    if configs.model_type == 1:
        target_u = torch.flatten(batch['target_u'].to(configs.training_device), 0, 1)

    episode_returns = batch['episode_return'][batch['done']]  # [Bool] equals masking

    with learn_lock:
        if configs.model_type == 0:
            Q = training_model.training_forward(obs_z, obs_x)
            loss = nn.functional.mse_loss(Q, target_R)
        elif configs.model_type == 1:  # separate Q_w/Q_l
            pr, Q_w, Q_l = training_model.training_forward(obs_z, obs_x)
            loss1 = nn.functional.binary_cross_entropy(pr, target_u)
            l_w = nn.functional.mse_loss(Q_w, target_R, reduction='none') * target_u
            l_l = nn.functional.mse_loss(Q_l, target_R, reduction='none') * (1. - target_u)
            loss2 = l_w.mean() + l_l.mean()
            loss = loss1 + loss2
        else:
            ValueError("model_type Error!")
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(training_model.parameters(), max_grad_norm)
        optim.step()
        for actor_models in actor_models_dict.values():
            actor_models.models[role].load_state_dict(training_model.state_dict())

    stats = {
        'mean_episode_return_' + role: torch.mean(episode_returns).item(),
        'loss_' + role: loss.detach().cpu().item(),
    }
    if configs.model_type != 0:
        stats['loss1_' + role] = loss1.detach().cpu().item(),  # pr loss
        stats['loss2_' + role] = loss2.detach().cpu().item(),  # Q' loss
    return stats  # mean_episode_return + loss


def training(configs):
    """
    This is the main function for training. It will first
    initialize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with multiple threads.
    """
    if not configs.actor_device_cpu or configs.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, "
                "please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")

    rootdir = os.path.expandvars(os.path.expanduser(configs.savedir))
    basepath = os.path.join(rootdir, configs.xpid)
    if not os.path.exists(basepath):
        print('Creating log directory:', basepath)
        os.makedirs(basepath, exist_ok=True)
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (configs.savedir, configs.xpid, 'model.pth')))

    T = configs.buffer_size
    B = configs.batch_buffer

    device_iter = []
    if configs.actor_device_cpu:  # actor_device_cpu means to use cpu as the actor device
        device_iter = ['cpu']
    else:
        gpu_devices = configs.gpu_devices.split(',')
        for i in range(len(gpu_devices)):
            device_iter.append('cuda:' + str(i))
            if gpu_devices[i] == configs.training_device:  # match gpu number
                configs.training_device = "cuda:" + str(i)
    if configs.training_device != 'cpu' and 'cuda:' not in configs.training_device:
        ValueError("training_device Mismatch!")

    actor_models_dict = {}  # dict[device].models[role]
    for device in device_iter:
        if configs.model_type == 0:
            actor_models = Models(device=device)
        elif configs.model_type == 1:
            actor_models = NewModels(device=device)
        else:
            ValueError("model_type Error!")
        actor_models.share_memory()
        actor_models.eval()
        actor_models_dict[device] = actor_models

    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
    full_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
    epsilons = [ctx.Value('d', configs.epsilon_greedy, lock=False) for _ in range(configs.num_actors)]  # variable
    max_grad_norm = configs.max_grad_norm

    # Learner model for training
    buffers = create_buffers_new(configs)
    if configs.model_type == 0:
        training_models = Models(device=configs.training_device)
    else:
        training_models = NewModels(device=configs.training_device)  # three models
    training_models.train()
    optimizers = create_optimizers(configs, training_models)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
    ]
    if configs.model_type != 0:
        stat_keys.extend(['loss1_landlord', 'loss2_landlord', 'loss1_landlord_up',
                          'loss2_landlord_up', 'loss1_landlord_down', 'loss2_landlord_down', ])
    frames, stats = 0, {k: 0. for k in stat_keys}
    role_frames = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}

    # Load models if any
    if configs.load_model:
        checkpoint_states = torch.load(checkpointpath, map_location=configs.training_device)
        for k in role_list:
            training_models.models[k].load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in device_iter:
                actor_models_dict[device].models[k].load_state_dict(training_models.models[k].state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        role_frames = checkpoint_states["role_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # Starting actor processes
    for device in device_iter:
        for i in range(configs.num_actors):  # launch actor processes
            actor = ctx.Process(
                target=act_proc_new,
                args=(i, device, free_queue, full_queue, actor_models_dict[device], buffers, configs, epsilons[i]))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, role, buffer_lock, learn_lock, lock=threading.Lock()):
        """ Learner Thread """
        nonlocal frames, role_frames, stats  # 嵌套函数的共享变量
        while True:
            batch = get_batch(free_queue[role], full_queue[role], buffers[role], configs, buffer_lock)
            _stats = update_loss_new(role, actor_models_dict, training_models.models[role], batch,
                                     optimizers[role], configs, learn_lock, max_grad_norm)
            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = {'frames': frames}
                to_log.update({k: stats[k] for k in stat_keys})
                frames += T * B
                role_frames[role] += T * B
        return

    # for device in device_iterator:
    for m in range(configs.num_buffers):
        free_queue['landlord'].put(m)
        free_queue['landlord_up'].put(m)
        free_queue['landlord_down'].put(m)

    threads = []
    buffer_locks = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}
    train_locks = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}

    for i in range(configs.num_threads):
        for role in role_list:  # launch learning threads
            thread = threading.Thread(
                target=batch_and_learn, name='batch-and-learn-%d' % i,
                args=(i, role, buffer_locks[role], train_locks[role]),
                daemon=True)
            thread.start()
            threads.append(thread)

    def checkpoint(frames):
        if not configs.disable_checkpoint:
            log.info('Saving checkpoint to %s', checkpointpath)
            _models = training_models.models  # {r:Model}
            torch.save({
                'model_state_dict': {k: _models[k].state_dict() for k in _models},
                'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
                "stats": stats,
                'configs': vars(configs),
                'frames': frames,
                'role_frames': role_frames
            }, checkpointpath)

            # Save the weights for evaluation purpose
            for role in role_list:
                file_name = role + '_' + str(frames) + '_' + str(role_frames[role]) + '.ckpt'
                with train_locks[role]:
                    model_weights_dir = os.path.expandvars(os.path.expanduser('%s/%s/%s' % (
                        configs.savedir, configs.xpid, file_name)))
                    torch.save(_models[role].state_dict(), model_weights_dir)

    savepoints = [5e7 * i for i in range(1, 100)]  # automatically save models in every 5e7 frames
    savepoint_counter = int(frames / 5e7)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - configs.save_interval * 60 / 2
        while True:
            start_frames = frames
            role_start_frames = {k: role_frames[k] for k in role_frames}
            start_time = timer()
            time.sleep(5.)
            if frames < configs.total_frames:
                if frames > savepoints[savepoint_counter]:
                    savepoint_counter += 1
                    checkpoint(frames)
                elif timer() - last_checkpoint_time > configs.save_interval * 60:
                    checkpoint(frames)
                    last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:  # 2min
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)
            role_fps = {k: (role_frames[k] - role_start_frames[k]) / (end_time - start_time)
                        for k in role_frames}
            log.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                     frames,
                     role_frames['landlord'],
                     role_frames['landlord_up'],
                     role_frames['landlord_down'],
                     fps,
                     fps_avg,
                     role_fps['landlord'],
                     role_fps['landlord_up'],
                     role_fps['landlord_down'],
                     pprint.pformat(stats))
            if not configs.forever_training and frames > configs.total_frames:
                break

    except KeyboardInterrupt:
        log.info("Main process ends: KeyboardInterrupt")
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    return 0
