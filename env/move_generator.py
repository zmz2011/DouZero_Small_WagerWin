from .utils import MIN_SINGLE_CARDS, MIN_PAIRS, MIN_TRIPLES, select
import collections
import itertools
from typing import List


def _gen_serial_moves(cards: List[int], min_serial: int, repeat: int = 1, repeat_num=0):  # 顺子
    if repeat_num < min_serial:  # at least repeat_num is min_serial
        repeat_num = 0

    # single_cards = sorted(list(set(cards)))
    seq_records = []
    moves = []
    if len(cards) < min_serial or len(cards) < repeat_num:
        return moves

    start = i = 0
    longest = 1
    while i < len(cards):
        if i + 1 < len(cards) and cards[i + 1] - cards[i] == 1:
            longest += 1
            i += 1
        else:
            seq_records.append((start, longest))
            i += 1
            start = i
            longest = 1

    for seq in seq_records:
        if seq[1] < min_serial:
            continue
        start, longest = seq[0], seq[1]
        longest_list = cards[start: start + longest]

        if repeat_num == 0:  # No limitation on how many sequences
            steps = min_serial
            while steps <= longest:
                index = 0
                while steps + index <= longest:
                    target_move = sorted(longest_list[index: index + steps] * repeat)
                    moves.append(target_move)
                    index += 1
                steps += 1
        else:  # repeat_num > 0
            if longest < repeat_num:
                continue
            index = 0
            while index + repeat_num <= longest:
                target_move = sorted(longest_list[index: index + repeat_num] * repeat)
                moves.append(target_move)
                index += 1

    return moves


class MovesGener(object):
    """
    This is for generating the possible combinations
    """

    def __init__(self, cards_list: List[int]):
        self.cards_list = sorted(cards_list)
        self.cards_dict = collections.defaultdict(int)
        for i in self.cards_list:
            self.cards_dict[i] += 1

        self.single_card_moves: List[List[int]] = []
        self.gen_type_1_single()
        self.pair_cards_moves: List[List[int]] = []
        self.gen_type_2_pair()
        self.triple_cards_moves: List[List[int]] = []
        self.gen_type_3_triple()
        self.bomb_moves: List[List[int]] = []
        self.gen_type_4_bomb()
        self.king_bomb_moves: List[List[int]] = []
        self.gen_type_5_king_bomb()

    def gen_type_1_single(self):
        # self.single_card_moves = []
        for k in self.cards_dict.keys():
            self.single_card_moves.append([k])
        return self.single_card_moves

    def gen_type_2_pair(self):
        # self.pair_cards_moves = []
        for k, v in self.cards_dict.items():
            if v >= 2:
                self.pair_cards_moves.append([k, k])
        return self.pair_cards_moves

    def gen_type_3_triple(self):
        # self.triple_cards_moves = []
        for k, v in self.cards_dict.items():
            if v >= 3:
                self.triple_cards_moves.append([k, k, k])
        return self.triple_cards_moves

    def gen_type_4_bomb(self):
        # self.bomb_moves = []
        for k, v in self.cards_dict.items():
            if v == 4:
                self.bomb_moves.append([k, k, k, k])
        return self.bomb_moves

    def gen_type_5_king_bomb(self):
        # self.king_bomb_moves = []
        if 20 in self.cards_dict and 30 in self.cards_dict:
            self.king_bomb_moves.append([20, 30])
        return self.king_bomb_moves

    def gen_type_6_3_1(self):
        result = []
        for i in self.triple_cards_moves:
            for t in self.single_card_moves:
                if t[0] != i[0]:
                    result.append(sorted(t + i))
        return result

    def gen_type_7_3_2(self):
        result = []
        for i in self.triple_cards_moves:
            for t in self.pair_cards_moves:
                if t[0] != i[0]:
                    result.append(sorted(t + i))
        return result

    def gen_type_8_serial_single(self, repeat_num=0):
        single_list = [i[0] for i in self.single_card_moves]
        return _gen_serial_moves(single_list, MIN_SINGLE_CARDS, repeat=1, repeat_num=repeat_num)

    def gen_type_9_serial_pair(self, repeat_num=0):
        pair_list = [i[0] for i in self.pair_cards_moves]
        return _gen_serial_moves(pair_list, MIN_PAIRS, repeat=2, repeat_num=repeat_num)

    def gen_type_10_serial_triple(self, repeat_num=0):
        single_triples = [i[0] for i in self.triple_cards_moves]
        return _gen_serial_moves(single_triples, MIN_TRIPLES, repeat=3, repeat_num=repeat_num)

    def gen_type_11_serial_3_1(self, repeat_num=0):  # todo
        serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
        serial_3_1_moves = list()

        for s3 in serial_3_moves:  # s3 is like [3,3,3,4,4,4]
            s3_set = set(s3)
            new_cards = list(set(self.cards_dict) - s3_set)
            subcards = select(new_cards, len(s3_set))  # Get any s3_len items from cards
            for i in subcards:
                serial_3_1_moves.append(s3 + i)

        return list(k for k, _ in itertools.groupby(serial_3_1_moves))  # todo:groupby

    # def gen_type_12_serial_3_2(self, repeat_num=0):
    #     serial_3_moves = self.gen_type_10_serial_triple(repeat_num=repeat_num)
    #     serial_3_2_moves = list()
    #     pair_set = sorted([k for k, v in self.cards_dict.items() if v >= 2])
    #
    #     for s3 in serial_3_moves:
    #         s3_set = set(s3)
    #         pair_candidates = [i for i in pair_set if i not in s3_set]
    #
    #         # Get any s3_len items from cards
    #         subcards = select(pair_candidates, len(s3_set))
    #         for i in subcards:
    #             serial_3_2_moves.append(sorted(s3 + i * 2))
    #
    #     return serial_3_2_moves

    def gen_type_13_4_2(self):
        result = list()
        for fc in self.bomb_moves:
            cards_list = [k for k in self.cards_list if k != fc[0]]
            subcards = select(cards_list, 2)
            for i in subcards:
                result.append(fc + i)
        return list(k for k, _ in itertools.groupby(result))  # todo:?

    # def gen_type_14_4_22(self):
    #     four_cards = list()
    #     for k, v in self.cards_dict.items():
    #         if v == 4:
    #             four_cards.append(k)
    #
    #     result = list()
    #     for fc in four_cards:
    #         cards_list = [k for k, v in self.cards_dict.items() if k != fc and v >= 2]
    #         subcards = select(cards_list, 2)
    #         for i in subcards:
    #             result.append([fc] * 4 + [i[0], i[0], i[1], i[1]])
    #     return result

    # generate all possible moves from given cards
    def gen_all_moves(self):
        moves = []
        moves.extend(self.single_card_moves)
        moves.extend(self.pair_cards_moves)
        moves.extend(self.triple_cards_moves)
        moves.extend(self.bomb_moves)
        moves.extend(self.king_bomb_moves)
        moves.extend(self.gen_type_6_3_1())
        moves.extend(self.gen_type_7_3_2())
        moves.extend(self.gen_type_8_serial_single())
        moves.extend(self.gen_type_9_serial_pair())
        moves.extend(self.gen_type_10_serial_triple())
        moves.extend(self.gen_type_11_serial_3_1())
        # moves.extend(self.gen_type_12_serial_3_2())
        moves.extend(self.gen_type_13_4_2())
        # moves.extend(self.gen_type_14_4_22())
        return moves
