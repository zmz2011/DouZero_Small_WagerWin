import os
from dmc import parser, training

if __name__ == '__main__':
    configs = parser.parse_args()
    if configs.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu_devices
    training(configs)
