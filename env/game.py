from typing import List, Dict
from copy import deepcopy
from .move_generator import MovesGener
from . import move_detector as md
from . import move_selector as ms

# EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
#                     8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
#                     13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}
#
# RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
#                     '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
#                     'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30}

bombs = [[3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6],
         [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9], [10, 10, 10, 10],
         [11, 11, 11, 11], [12, 12, 12, 12], [13, 13, 13, 13], [14, 14, 14, 14],
         [17, 17, 17, 17], [20, 30]]


class InfoSet(object):
    """
    The game state is described as infoset, which
    includes all the information in the current situation,
    such as the hand cards of the three players, the
    historical moves, etc.
    """

    def __init__(self, player_role):
        # The player position, i.e., landlord, landlord_down, or landlord_up
        self.player_role = player_role
        # The hand cards of the current player. A list.
        self.player_hand_cards: List[int] = None
        # The number of cards left for each player. It is a dict with str-->int
        self.num_cards_left_dict: Dict[str, int] = None
        # The three landlord cards. A list.
        self.landlord_public_cards: List[int] = None
        # The historical moves. It is a list of list
        self.action_seq: List[List] = []
        # The union of the hand cards of the other two players for the current player
        self.other_hand_cards: List[int] = None
        # The legal actions for the current move. It is a list of list
        self.legal_actions: List[List] = None
        # The most recent valid move
        self.last_move: List = None
        # The most recent two moves
        self.last_two_moves: List = None
        # The last moves for all roles
        self.last_move_dict = None
        # The played cards so far. It is a list.
        self.played_cards: List[int] = None
        # The hand cards of all the players. It is a dict.
        self.all_handcards: Dict[str, List] = None
        # Last player position that plays a valid move, i.e., not `pass`
        self.last_valid_pid: str = None
        # The number of bombs played so far
        self.bomb_played: int = None


class GameEnv(object):

    def __init__(self, players: Dict):

        self.action_seq = []

        self.landlord_public_cards: List[int] = None
        self.game_over: bool = False

        self.acting_player_role: str = None
        self.player_utility_dict = None

        self.players = players  # {role: DummyAgent}

        self.last_move_dict = {'landlord': [], 'landlord_up': [], 'landlord_down': []}

        self.played_cards = {'landlord': [], 'landlord_up': [], 'landlord_down': []}

        self.last_move = []
        self.last_two_moves = []

        self.num_wins = {'landlord': 0, 'farmer': 0}
        self.num_scores = {'landlord': 0, 'farmer': 0}
        self.game_nums = 0

        self.info_sets = {'landlord': InfoSet('landlord'),
                          'landlord_up': InfoSet('landlord_up'),
                          'landlord_down': InfoSet('landlord_down')}

        self.bomb_played = 0
        self.last_valid_pid = 'landlord'

    def card_play_init(self, card_play_data):
        self.info_sets['landlord'].player_hand_cards = card_play_data['landlord']
        self.info_sets['landlord_up'].player_hand_cards = card_play_data['landlord_up']
        self.info_sets['landlord_down'].player_hand_cards = card_play_data['landlord_down']
        self.landlord_public_cards = card_play_data['landlord_public_cards']
        self.acting_player_role = 'landlord'
        self.game_infoset = self.get_infoset()

    def game_done(self):
        if len(self.info_sets['landlord'].player_hand_cards) == 0 or \
                len(self.info_sets['landlord_up'].player_hand_cards) == 0 or \
                len(self.info_sets['landlord_down'].player_hand_cards) == 0:
            self.compute_player_utility()
            self.update_num_wins_scores()
            self.game_nums += 1
            self.game_over = True

    def compute_player_utility(self):
        if len(self.info_sets['landlord'].player_hand_cards) == 0:
            self.player_utility_dict = {'landlord': 2, 'farmer': -1}
        else:
            self.player_utility_dict = {'landlord': -2, 'farmer': 1}

    def update_num_wins_scores(self):
        for role, utility in self.player_utility_dict.items():
            base_score = 2 if role == 'landlord' else 1
            if utility > 0:
                self.num_wins[role] += 1
                self.winner = role
                self.num_scores[role] += base_score * (2 ** self.bomb_played)
            else:
                self.num_scores[role] -= base_score * (2 ** self.bomb_played)

    def get_winner(self):
        return self.winner

    def get_bomb_played(self):
        return self.bomb_played

    def step(self):
        action = self.players[self.acting_player_role].act(self.game_infoset)
        assert action in self.game_infoset.legal_actions

        if len(action) > 0:
            self.last_valid_pid = self.acting_player_role

        if action in bombs:
            self.bomb_played += 1

        self.last_move_dict[self.acting_player_role] = action.copy()

        self.action_seq.append(action)
        self.update_acting_player_hand_cards(action)

        self.played_cards[self.acting_player_role] += action

        if self.acting_player_role == 'landlord' and len(action) > 0 and len(self.landlord_public_cards) > 0:
            for card in action:
                if card in self.landlord_public_cards:
                    self.landlord_public_cards.remove(card)

        self.game_done()
        if not self.game_over:
            self.get_next_player_role()
            self.game_infoset = self.get_infoset()

    def get_last_move(self):
        last_move = []
        if len(self.action_seq) != 0:
            if len(self.action_seq[-1]) == 0:
                last_move = self.action_seq[-2]
            else:
                last_move = self.action_seq[-1]
        return last_move

    def get_last_two_moves(self):
        last_two_moves = [[], []]
        for card in self.action_seq[-2:]:
            last_two_moves.insert(0, card)
        last_two_moves = last_two_moves[:2]
        return last_two_moves

    def get_next_player_role(self):
        if self.acting_player_role is None:
            self.acting_player_role = 'landlord'
        else:
            if self.acting_player_role == 'landlord':
                self.acting_player_role = 'landlord_down'
            elif self.acting_player_role == 'landlord_down':
                self.acting_player_role = 'landlord_up'
            else:
                self.acting_player_role = 'landlord'
        return self.acting_player_role

    def update_acting_player_hand_cards(self, action):
        if action:
            for card in action:
                self.info_sets[self.acting_player_role].player_hand_cards.remove(card)
            self.info_sets[self.acting_player_role].player_hand_cards.sort()

    def get_legal_card_play_actions(self):
        mg = MovesGener(self.info_sets[self.acting_player_role].player_hand_cards)
        action_sequence = self.action_seq

        rival_move = []
        if len(action_sequence) != 0:
            if len(action_sequence[-1]) == 0:
                rival_move = action_sequence[-2]
            else:
                rival_move = action_sequence[-1]

        rival_type = md.get_move_type(rival_move)
        rival_move_type = rival_type['type']
        rival_move_len = rival_type.get('len', 1)
        moves = list()

        if rival_move_type == md.TYPE_0_PASS:
            moves = mg.gen_all_moves()

        elif rival_move_type == md.TYPE_1_SINGLE:
            all_moves = mg.single_card_moves
            moves = ms.filter_type_1_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_2_PAIR:
            all_moves = mg.pair_cards_moves
            moves = ms.filter_type_2_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_3_TRIPLE:
            all_moves = mg.triple_cards_moves
            moves = ms.filter_type_3_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_4_BOMB:
            all_moves = mg.bomb_moves + mg.king_bomb_moves
            moves = ms.filter_type_4_bomb(all_moves, rival_move)

        elif rival_move_type == md.TYPE_5_KING_BOMB:
            moves = []

        elif rival_move_type == md.TYPE_6_3_1:
            all_moves = mg.gen_type_6_3_1()
            moves = ms.filter_type_6_3_1(all_moves, rival_move)

        elif rival_move_type == md.TYPE_7_3_2:
            all_moves = mg.gen_type_7_3_2()
            moves = ms.filter_type_7_3_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_8_SERIAL_SINGLE:
            all_moves = mg.gen_type_8_serial_single(repeat_num=rival_move_len)
            moves = ms.filter_type_8_serial_single(all_moves, rival_move)

        elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
            all_moves = mg.gen_type_9_serial_pair(repeat_num=rival_move_len)
            moves = ms.filter_type_9_serial_pair(all_moves, rival_move)

        elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
            all_moves = mg.gen_type_10_serial_triple(repeat_num=rival_move_len)
            moves = ms.filter_type_10_serial_triple(all_moves, rival_move)

        elif rival_move_type == md.TYPE_11_SERIAL_3_1:
            all_moves = mg.gen_type_11_serial_3_1(repeat_num=rival_move_len)
            moves = ms.filter_type_11_serial_3_1(all_moves, rival_move)

        # elif rival_move_type == md.TYPE_12_SERIAL_3_2:
        #     all_moves = mg.gen_type_12_serial_3_2(repeat_num=rival_move_len)
        #     moves = ms.filter_type_12_serial_3_2(all_moves, rival_move)

        elif rival_move_type == md.TYPE_13_4_2:
            all_moves = mg.gen_type_13_4_2()
            moves = ms.filter_type_13_4_2(all_moves, rival_move)

        # elif rival_move_type == md.TYPE_14_4_22:
        #     all_moves = mg.gen_type_14_4_22()
        #     moves = ms.filter_type_14_4_22(all_moves, rival_move)

        if rival_move_type not in [md.TYPE_0_PASS, md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB]:
            moves = moves + mg.bomb_moves + mg.king_bomb_moves

        if len(rival_move) != 0:  # rival_move is not 'pass'
            moves.extend([[]])

        # for m in moves:
        #     m.sort()

        return moves

    def reset(self):
        self.action_seq = []

        self.landlord_public_cards = None
        self.game_over = False

        self.acting_player_role = None
        self.player_utility_dict = None

        self.last_move_dict = {'landlord': [],
                               'landlord_up': [],
                               'landlord_down': []}

        self.played_cards = {'landlord': [],
                             'landlord_up': [],
                             'landlord_down': []}

        self.last_move = []
        self.last_two_moves = []

        self.info_sets = {'landlord': InfoSet('landlord'),
                          'landlord_up': InfoSet('landlord_up'),
                          'landlord_down': InfoSet('landlord_down')}

        self.bomb_played = 0
        self.last_valid_pid = 'landlord'

    def get_infoset(self):
        self.info_sets[self.acting_player_role].last_valid_pid = self.last_valid_pid
        self.info_sets[self.acting_player_role].legal_actions = self.get_legal_card_play_actions()
        self.info_sets[self.acting_player_role].bomb_played = self.bomb_played
        self.info_sets[self.acting_player_role].last_move = self.get_last_move()
        self.info_sets[self.acting_player_role].last_two_moves = self.get_last_two_moves()
        self.info_sets[self.acting_player_role].last_move_dict = self.last_move_dict
        self.info_sets[self.acting_player_role].num_cards_left_dict = \
            {pos: len(self.info_sets[pos].player_hand_cards) for pos in ['landlord', 'landlord_up', 'landlord_down']}

        self.info_sets[self.acting_player_role].other_hand_cards = []
        for role in ['landlord', 'landlord_up', 'landlord_down']:
            if role != self.acting_player_role:
                self.info_sets[self.acting_player_role].other_hand_cards += \
                    self.info_sets[role].player_hand_cards

        self.info_sets[self.acting_player_role].played_cards = self.played_cards
        self.info_sets[self.acting_player_role].landlord_public_cards = self.landlord_public_cards
        self.info_sets[self.acting_player_role].action_seq = self.action_seq
        self.info_sets[self.acting_player_role].all_handcards = \
            {pos: self.info_sets[pos].player_hand_cards for pos in ['landlord', 'landlord_up', 'landlord_down']}

        return deepcopy(self.info_sets[self.acting_player_role])