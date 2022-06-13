from collections import Counter
import numpy as np

from .game import GameEnv

Card2Column = {6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6, 13: 7, 14: 8}  # reduced game
# Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 17: 12}
NumOnes2Array = {0: np.array([0., 0., 0., 0.], dtype=np.float32),
                 1: np.array([1., 0., 0., 0.], dtype=np.float32),
                 2: np.array([1., 1., 0., 0.], dtype=np.float32),
                 3: np.array([1., 1., 1., 0.], dtype=np.float32),
                 4: np.array([1., 1., 1., 1.], dtype=np.float32)}

deck = []
# change card nums to (short) 6~A
for i in range(6, 15):
    deck.extend([i for _ in range(4)])
# deck.extend([17 for _ in range(4)])
deck.extend([20, 30])


class Env:
    """
    Doudizhu multi-agent wrapper
    """

    def __init__(self, objective="adp"):
        """
        Objective is wp/adp/logadp. It indicates whether considers
        bomb in reward calculation. Here, we use dummy agents.
        This is because, in the original game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """
        self.objective = objective

        # Initialize players
        # We use three dummy player for the target position
        self.players = {}
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            self.players[position] = DummyAgent(position)

        # Initialize the internal environment
        self._env = GameEnv(self.players)

        self.infoset = None

    def reset(self, card_play_data=None):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        self._env.reset()

        # Randomly shuffle the deck
        _deck = deck.copy()
        np.random.shuffle(_deck)
        # change card nums 12+12+14 (short)
        if card_play_data is None:
            card_play_data = {'landlord': _deck[:14],
                              'landlord_up': _deck[14:26],
                              'landlord_down': _deck[26:38],
                              'landlord_public_cards': _deck[12:14],
                              }  # reduced game
            for key in card_play_data:
                card_play_data[key].sort()

        # Initialize the cards
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return get_obs_new(self.infoset)

    def step(self, action):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next observation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
        """
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = self._get_reward()
            obs = None
        else:
            obs = get_obs_new(self.infoset)
        return obs, reward, done, {'bomb_played': self.infoset.bomb_played}

    def _get_reward(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        winner = self._game_winner
        bomb_num = self._game_bomb_num
        if winner == 'landlord':
            if self.objective == 'adp':
                return 2.0 ** bomb_num
            elif self.objective == 'logadp':
                return bomb_num + 1.0
            else:
                return 1.0
        else:
            if self.objective == 'adp':
                return -2.0 ** bomb_num
            elif self.objective == 'logadp':
                return -bomb_num - 1.0
            else:
                return -1.0

    @property
    def _game_infoset(self):
        """
        Here, infoset is defined as all the information
        in the current situation, including the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perfect information. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """
        The number of bombs played so far. This is used as
        a feature of the neural network and is also used to
        calculate ADP.
        """
        return self._env.get_bomb_played()

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_role

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over


class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """

    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action


def _get_one_hot_array(num, max_nums):
    """
    A utility function to obtain one-hot encoding
    """
    one_hot = np.zeros(max_nums, dtype=np.float32)
    one_hot[num - 1] = 1

    return one_hot


def cards2array(list_cards):
    if len(list_cards) == 0:
        return np.zeros(38, dtype=np.float32)
    matrix = np.zeros((9, 4), dtype=np.float32)
    jokers = np.zeros(2, dtype=np.float32)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1.
        elif card == 30:
            jokers[1] = 1.
    return np.concatenate((matrix.flatten(), jokers))


def _action_seq_list2array(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 38), dtype=np.float32)
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(4, 114)  # change from 5 to 4
    return action_seq_array


def _process_action_seq(sequence, length=12):  # 15->12
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence


def _get_one_hot_bomb(bomb_num):  # reduce to 11
    return _get_one_hot_array(bomb_num + 1, 11)


def get_obs_new(infoset):
    if infoset.player_role == 'landlord':
        return _get_obs_landlord_new(infoset)
    elif infoset.player_role == 'landlord_up':
        return _get_obs_landlord_up_new(infoset)
    elif infoset.player_role == 'landlord_down':
        return _get_obs_landlord_down_new(infoset)
    else:
        raise ValueError('get_obs_new() error')


def _get_obs_landlord_new(infoset):
    num_legal_actions = len(infoset.legal_actions)

    my_handcards = cards2array(infoset.player_hand_cards)
    other_handcards = cards2array(infoset.other_hand_cards)
    last_action = cards2array(infoset.last_move)
    landlord_up_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict['landlord_up'], 12)
    landlord_down_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict['landlord_down'], 12)
    landlord_up_played_cards = cards2array(infoset.played_cards['landlord_up'])
    landlord_down_played_cards = cards2array(infoset.played_cards['landlord_down'])
    bomb_num = _get_one_hot_bomb(infoset.bomb_played)

    my_action_batch = np.zeros((num_legal_actions, 38), dtype=np.float32)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j] = cards2array(action)

    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,
                             bomb_num))
    x_batch = np.repeat(x_no_action[np.newaxis, :], num_legal_actions, axis=0)
    x_batch = np.hstack((x_batch, my_action_batch))
    z = _action_seq_list2array(_process_action_seq(infoset.action_seq))

    obs = {
        'position': 'landlord',
        'x_batch': x_batch,
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action,
        'z': z,
    }
    return obs


def _get_obs_landlord_up_new(infoset):
    num_legal_actions = len(infoset.legal_actions)

    my_handcards = cards2array(infoset.player_hand_cards)
    other_handcards = cards2array(infoset.other_hand_cards)
    last_action = cards2array(infoset.last_move)
    last_landlord_action = cards2array(infoset.last_move_dict['landlord'])
    landlord_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict['landlord'], 14)
    landlord_played_cards = cards2array(infoset.played_cards['landlord'])
    last_teammate_action = cards2array(infoset.last_move_dict['landlord_down'])
    teammate_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict['landlord_down'], 12)
    teammate_played_cards = cards2array(infoset.played_cards['landlord_down'])
    bomb_num = _get_one_hot_bomb(infoset.bomb_played)

    my_action_batch = np.zeros((num_legal_actions, 38), dtype=np.float32)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j] = cards2array(action)

    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    x_batch = np.repeat(x_no_action[np.newaxis, :], num_legal_actions, axis=0)
    x_batch = np.hstack((x_batch, my_action_batch))

    z = _action_seq_list2array(_process_action_seq(infoset.action_seq))

    obs = {
        'position': 'landlord_up',
        'x_batch': x_batch,
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action,
        'z': z,
    }
    return obs


def _get_obs_landlord_down_new(infoset):
    num_legal_actions = len(infoset.legal_actions)

    my_handcards = cards2array(infoset.player_hand_cards)
    other_handcards = cards2array(infoset.other_hand_cards)
    last_action = cards2array(infoset.last_move)
    last_landlord_action = cards2array(infoset.last_move_dict['landlord'])
    landlord_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict['landlord'], 14)
    landlord_played_cards = cards2array(infoset.played_cards['landlord'])
    last_teammate_action = cards2array(infoset.last_move_dict['landlord_up'])
    teammate_num_cards_left = _get_one_hot_array(infoset.num_cards_left_dict['landlord_up'], 12)
    teammate_played_cards = cards2array(infoset.played_cards['landlord_up'])
    bomb_num = _get_one_hot_bomb(infoset.bomb_played)

    my_action_batch = np.zeros((num_legal_actions, 38), dtype=np.float32)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j] = cards2array(action)

    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    x_batch = np.repeat(x_no_action[np.newaxis, :], num_legal_actions, axis=0)
    x_batch = np.hstack((x_batch, my_action_batch))
    z = _action_seq_list2array(_process_action_seq(infoset.action_seq))

    obs = {
        'position': 'landlord_down',
        'x_batch': x_batch,
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action,
        'z': z,
    }
    return obs
