import multiprocessing as mp
import pickle

import numpy as np

from env import GameEnv
from .deep_agent import *


def load_card_play_model(model_type, path=None, flags=None):
    player = None
    if model_type == 'rlcard':
        from rlcard_agent import RLCardAgent
        player = RLCardAgent()
    elif model_type == 'random':
        from random_agent import RandomAgent
        player = RandomAgent()
    elif model_type == 'landlordlstm':
        player = LandlordLstmAgent(path)
    elif model_type == 'peasantlstm':
        player = PeasantLstmAgent(path)
    elif model_type == 'newlandlordlstm':
        player = NewLandlordLstmAgent(path)
    elif model_type == 'newpeasantlstm':
        player = NewPeasantLstmAgent(path)
    return player


def mp_simulate(card_play_data_list, configs, q, flags):
    players = {}
    players['landlord'] = load_card_play_model(configs['landlord_type'], configs['landlord_path'], flags)
    players['landlord_up'] = load_card_play_model(configs['landlord_up_type'], configs['landlord_up_path'], flags)
    players['landlord_down'] = load_card_play_model(configs['landlord_down_type'], configs['landlord_down_path'], flags)

    env = GameEnv(players)
    print(len(card_play_data_list))
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.reset()

    q.put((env.num_wins['landlord'],
           env.num_wins['farmer'],
           env.num_scores['landlord'],
           env.num_scores['farmer'],
           ))


def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for _ in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)
    return card_play_data_list_each_worker


def evaluate(configs, flags):
    with open(configs['eval_data_path'], 'rb') as f:
        card_play_data_list = pickle.load(f)
    print(configs['landlord_path'])
    print(configs['landlord_up_path'])
    print(configs['landlord_down_path'])
    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, configs['num_workers'])

    num_landlord_wins = 0
    num_peasants_wins = 0
    num_landlord_scores = 0
    num_peasants_scores = 0

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for card_paly_data in card_play_data_list_each_worker:
        p = ctx.Process(
            target=mp_simulate,
            args=(card_paly_data, configs, q, flags))
        p.start()
        processes.append(p)

    # print("start evaluating")
    for p in processes:
        p.join()

    for i in range(configs['num_workers']):
        result = q.get()
        num_landlord_wins += result[0]
        num_peasants_wins += result[1]
        num_landlord_scores += result[2]
        num_peasants_scores += result[3]

    num_total = num_landlord_wins + num_peasants_wins
    print("total:", num_total, "-", len(card_play_data_list))
    print('WP results:')
    print('landlord : peasants  -  {} : {}'.format(num_landlord_wins / num_total, num_peasants_wins / num_total))
    print('ADP results:')
    print(
        'landlord : peasants  -  {} : {}'.format(num_landlord_scores / num_total, 2 * num_peasants_scores / num_total))
