import argparse

parser = argparse.ArgumentParser(description='DouZero with new targets')

# General Settings
parser.add_argument('--xpid', default='x',
                    help='Experiment id')
parser.add_argument('--save_interval', default=600, type=int,
                    help='Time interval (in minutes) at which to save the model')
parser.add_argument('--objective', default='adp', type=str, choices=['adp', 'wp', 'logadp'],
                    help='Use ADP or WP as reward (default: ADP)')
parser.add_argument('--forever_training', action='store_true',
                    help='continue training after total_frames')

# Training settings
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as the actor device')
parser.add_argument('--gpu_devices', default='0', type=str,  # '0'/'0,1'...
                    help='Which GPUs to be used for this program')
parser.add_argument('--num_actors', default=4, type=int,
                    help='The number of actor processes for each actor device')
parser.add_argument('--training_device', default='0', type=str,  # single device
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model if exists')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='checkpoints',
                    help='Root dir where experiment data will be saved')

# Hyperparameters
parser.add_argument('--total_frames', default=int(3e9), type=int,
                    help='Total timesteps for model training')
parser.add_argument('--model_type', default=1, type=int,  #
                    help='type of model: 0: vallina douzero, 1: new framework')
parser.add_argument('--epsilon_greedy', default=0.01, type=float,
                    help='The prob for exploration')
parser.add_argument('--batch_buffer', default=24, type=int,
                    help='Learner`s batch_size = batch_buffer * buffer_size')
parser.add_argument('--buffer_size', default=128, type=int,
                    help='The unroll length and buffer size')
parser.add_argument('--num_buffers', default=64, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=1, type=int,  # 1 is enough
                    help='Number of learner threads')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='Max norm of gradients')

# Optimizer settings
parser.add_argument('--learning_rate', default=1e-4, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--eps', default=1e-6, type=float,
                    help='RMSProp epsilon')
