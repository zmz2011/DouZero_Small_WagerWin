import os
import argparse

from evaluation.simulation import evaluate


def get_configs():
    configs = {}
    # random/rlcard/landlordlstm/peasantlstm/newlandlordlstm/newpeasantlstm
    configs['landlord_type'] = 'newlandlordlstm'
    configs['landlord_up_type'] = 'newpeasantlstm'
    configs['landlord_down_type'] = 'newpeasantlstm'
    configs['landlord_path'] = '.'
    configs['landlord_up_path'] = '.'
    configs['landlord_down_path'] = '.'
    configs['eval_data_path'] = 'eval_data.pkl'
    configs['num_workers'] = 4
    return configs


parser = argparse.ArgumentParser(description='DouZero with new loss: Evaluation')

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    flags = parser.parse_args()
    configs = get_configs()
    evaluate(configs, flags)
