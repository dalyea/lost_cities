#!/usr/bin/env python
# coding: utf-8


# LOST CITIES
# General setup for 18 card game
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random
import pandas as pd
from collections import defaultdict
import pprint
import datetime


# GAME SETUP
COLORS = ['R', 'B', 'G']
NUMBERS = ['X', '2', '3', '4', '5', '6']
CARD_TO_IDX = {color + num: idx for idx, (color, num) in enumerate([(c, n) for c in COLORS for n in NUMBERS])}
draw_to_index = {draw: i for i, draw in enumerate(['deck', 'R', 'B', 'G'])}
tgt_pts = 7
color_cnt=len(COLORS)
per_color=len(NUMBERS)
card_cnt=len(COLORS)*len(NUMBERS)
state_size = 43


# BASICS & CLASSES
# Deck creation
def create_deck():
    deck = [color + num for color in COLORS for num in NUMBERS]
    random.shuffle(deck)
    return deck

# Environment Class
class LostCitiesEnv:
    def __init__(self):
        self.reset()
        # self.last_discard = None

    def reset(self):
        self.deck = create_deck()
        self.hands = {f'P{i+1}': [self.deck.pop() for _ in range(3)] for i in range(2)}
        self.expeditions = {player: defaultdict(list) for player in self.hands}
        self.center_piles = {color: [] for color in COLORS}
        self.players = list(self.hands.keys())
        self.current_player_idx = 0
        self.done = False
        # self.last_discard = None
        self.last_discards = {player: None for player in self.players}  # ✅ per-player discard tracking
        return self.get_state()

    def get_state(self):
        state = {
            'hands': {p: sorted(self.hands[p]) for p in self.hands},
            'expeditions': {p: {color: list(cards) for color, cards in self.expeditions[p].items()} for p in self.expeditions},
            'center': {color: list(self.center_piles[color]) for color in COLORS},
            'deck_size': len(self.deck),
            'current_player': self.players[self.current_player_idx]
        }
        return state

    def can_play_to_expedition(self, expedition, card):
        if not expedition:
            return True
        if card[1] == 'X':
            return False
        existing_values = [int(c[1]) for c in expedition if c[1] != 'X']
        last_number = max(existing_values) if existing_values else None
        if last_number is None:
            return True
        return int(card[1]) >= last_number

    def get_legal_actions(self, player):
        hand = self.hands[player]
        expeditions = self.expeditions[player]
        deck_size = len(self.deck)
    
        playable_to_expedition = []
        playable_to_center = []

        for color in COLORS:
            # Number cards (non-'X') in hand
            color_number_cards = [card for card in hand if card[0] == color and card[1] != 'X']
            expedition = expeditions[color]
            existing_numbers = [int(c[1]) for c in expedition if c[1] != 'X']
            highest_played = max(existing_numbers) if existing_numbers else 0
            valid_cards = [card for card in color_number_cards if int(card[1]) >= highest_played]
        
            # Apply lowest-card rule if enough deck remains
            if valid_cards:
                deck_remaining = len(self.deck)
                if deck_remaining >= 3:
                    min_val = min(int(c[1]) for c in valid_cards)
                    valid_cards = [c for c in valid_cards if int(c[1]) == min_val]
        
                playable_to_expedition.extend(valid_cards)
        
            # Multipliers (only if no numbers yet in expedition) - but maybe no other same color cards
            # multiplier_cards = [card for card in hand if card[0] == color and card[1] == 'X']
            # if multiplier_cards and not existing_numbers:
            #     playable_to_expedition.extend(multiplier_cards)

            # Multipliers (only if no numbers yet in expedition) - but must have at least one same color card
            multiplier_cards = [card for card in hand if card[0] == color and card[1] == 'X']
            if multiplier_cards and not existing_numbers:
                # Check for sum of same color cards 4+
                number_points = sum(int(c[1]) for c in hand if c[0] == color and c[1] != 'X')
                if number_points >= 4:
                    playable_to_expedition.extend(multiplier_cards)
                # Check for presence of at least one number card 4+
                # has_decent_number = any(int(c[1]) >= 4 for c in hand if c[0] == color and c[1] != 'X')
                # if has_decent_number:
                #     playable_to_expedition.extend(multiplier_cards)


        if random.random() < 1e-8:
            if not playable_to_expedition:
                print(f"\n[DEBUG] No playable expedition cards for player {player}")
                print(f"Hand: {sorted(hand)}")
                print(f"Expeditions:")
                for c in COLORS:
                    exp_cards = expeditions[c]
                    print(f"  {c}: {exp_cards}")
                print(f"Deck remaining: {len(self.deck)}\n")
                
        # ✅ Multipliers: Only if expedition has no numbers yet
        for card in hand:
            if card[1] == 'X':
                expedition = expeditions[card[0]]
                existing_numbers = [c for c in expedition if c[1] != 'X']
                if not existing_numbers:
                    playable_to_expedition.append(card)
                
        # ✅ Discards: Always legal to discard any card
        # for card in hand:
        #     playable_to_center.append(card)

        # Modification - do not play to center a card which could be put on an open exp
        for card in hand:
            expedition_pile = expeditions[card[0]]
            existing_numbers = [int(c[1]) for c in expedition_pile if c[1] != 'X']
            if existing_numbers:
                top_val = max(existing_numbers)
                card_val = int(card[1]) if card[1] != 'X' else None
                if card_val is not None and card_val >= top_val:
                    continue  # Don't allow discard—card is playable
            playable_to_center.append(card)
    
        actions = [("expedition", card) for card in playable_to_expedition] + [("center", card) for card in playable_to_center]
    
        # ✅ Drawing logic with redraw rule
        draws = ['deck'] + [c for c in COLORS if self.center_piles[c]]
        player_last_discard = self.last_discards.get(player, None)
        if player_last_discard is not None:
            discard_color, discard_card = player_last_discard
            if self.center_piles[discard_color] and self.center_piles[discard_color][-1] == discard_card:
                if discard_color in draws:
                    draws.remove(discard_color)

        # Filter out existing draws
        filtered_draws = []

        # New legal block, drawing an X when already started color
        deck_remaining = len(self.deck)
        
        for d in draws:
            if d == 'deck':
                filtered_draws.append(d)
            elif deck_remaining < 6:
                # Allow any center draws when deck is low
                filtered_draws.append(d)
            else:
                center_pile = self.center_piles[d]
                if center_pile:
                    top_card = center_pile[-1]
                    if top_card[1] == 'X':
                        expedition_started = bool(self.expeditions[player][d])  # expedition already started
                        if not expedition_started:
                            filtered_draws.append(d)
                        # else → skip this draw: can't use multiplier once expedition started
                    else:
                        filtered_draws.append(d)

        # Now reassign
        draws=filtered_draws

        return actions, draws

    def step(self, action, draw_choice):
        if self.done:
            return self.get_state(), 0, True
    
        player = self.players[self.current_player_idx]
        action_type, card = action
    
        # ✅ Remove card from hand BEFORE anything else
        assert card in self.hands[player], f"Player {player} does not have card {card}!"
        self.hands[player].remove(card)
    
        # ✅ Play action: place card on expedition or center pile
        if action_type == 'expedition':
            self.expeditions[player][card[0]].append(card)
            self.last_discards[player] = None
        elif action_type == 'center':
            self.center_piles[card[0]].append(card)
            self.last_discards[player] = (card[0], card)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
        # ✅ Draw phase (only happens ONCE per turn)
        if draw_choice == 'deck':
            if self.deck:
                self.hands[player].append(self.deck.pop())
            else:
                # No deck left, but should trigger game over below
                pass
        elif draw_choice in self.center_piles and self.center_piles[draw_choice]:
            self.hands[player].append(self.center_piles[draw_choice].pop())
        else:
            raise ValueError(f"Invalid draw_choice: {draw_choice}")
    
        # ✅ Hand must return to exactly 3 cards
        assert len(self.hands[player]) == 3, f"{player} has {len(self.hands[player])} cards in hand!"
    
        # ✅ Check for end of game (deck empty)
        if not self.deck:
            self.done = True
            reward = self.compute_score(player)
            return self.get_state(), reward, True
    
        # === ASSERTIONS ===

        # 1. Each player must have exactly 3 cards
        for p in self.players:
            assert len(self.hands[p]) == 3, f"{p} has {len(self.hands[p])} cards!"
    
        # 2. Each center pile must have at most 6 cards (can't discard more than that in 18-card game)
        for color, pile in self.center_piles.items():
            assert len(pile) <= per_color, f"Center pile {color} has {len(pile)} cards!"
    
        # 3. Each expedition pile must have at most 6 cards (max number per color)
        for player_exped in self.expeditions.values():
            for color, pile in player_exped.items():
                assert len(pile) <= per_color, f"Expedition {color} has {len(pile)} cards!"
            
        # ✅ Card count integrity check (should always be card_cnt)
        total_cards = sum(len(h) for h in self.hands.values()) + \
                      sum(len(p) for p in self.center_piles.values()) + \
                      sum(len(pile) for player_piles in self.expeditions.values() for pile in player_piles.values()) + \
                      len(self.deck)
        assert total_cards == card_cnt, f"Card count mismatch! Total cards: {total_cards}"

        # Validate expedition state for current player
        player_exped = self.expeditions[player]
        
        # Get list of expedition colors where at least one card has been played
        started_expeditions = [color for color, pile in player_exped.items() if pile]
        
        # Assert: Never more than 3 expeditions started
        assert len(started_expeditions) <= color_cnt, f"Too many expeditions started: {started_expeditions}"
        
        # Assert: No duplicate colors (guaranteed by defaultdict keys, but we double-check)
        assert len(started_expeditions) == len(set(started_expeditions)), f"Duplicate expedition colors: {started_expeditions}"

        # ✅ Switch to next player
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

        return self.get_state(), 0, False

    def compute_score(self, player):
        total = 0
        for color, cards in self.expeditions[player].items():
            if cards:
                values = [int(card[1]) for card in cards if card[1] != 'X']
                multiplier = 1 + sum(1 for card in cards if card[1] == 'X')
                expedition_score = multiplier * (sum(values) - tgt_pts)
                total += expedition_score
        return total

def summarize_rule_firings():
    return pprint.pformat(dict(rule_counter))


# 43 Features Setup
def extract_features(state):
    current_player = state['current_player']
    other_player = [p for p in state['hands'] if p != current_player][0]

    # Hand cards (18)
    hand_counts = np.zeros(len(CARD_TO_IDX))
    for card in state['hands'][current_player]:
        idx = CARD_TO_IDX[card]
        hand_counts[idx] += 1

    # Player expeditions (9 features)
    player_exped = []
    for color in COLORS:
        cards = state['expeditions'][current_player].get(color, [])
        values = [int(c[1]) for c in cards if c[1] != 'X']
        total = sum(values)
        multiplier = int(any(c[1] == 'X' for c in cards))
        count = len(cards)
        player_exped.extend([total, multiplier, count])

    # Discard piles (6)
    discard_info = []
    for color in COLORS:
        pile = state['center'][color]
        top_card = pile[-1][1] if pile else '0'
        top = int(top_card) if top_card != 'X' else 0
        # top = int(pile[-1][1]) if pile else 0
        count = len(pile)
        discard_info.extend([top, count])

    # Opponent expeditions (9 features)
    opp_exped = []
    for color in COLORS:
        cards = state['expeditions'][other_player].get(color, [])
        values = [int(c[1]) for c in cards if c[1] != 'X']
        total = sum(values)
        multiplier = int(any(c[1] == 'X' for c in cards))
        count = len(cards)
        opp_exped.extend([total, multiplier, count])

    # Deck size (1)
    deck_norm = state['deck_size'] / card_cnt

    return np.concatenate([hand_counts, player_exped, discard_info, opp_exped, [deck_norm]])


# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, draw_size, nn_layer_1, nn_layer_2, nn_layer_2_dropout):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, nn_layer_1) 
        self.fc2 = nn.Linear(nn_layer_1, nn_layer_2)         
        self.dropout = nn.Dropout(p=nn_layer_2_dropout)   

        # super(ActorCritic, self).__init__()
        # self.fc1 = nn.Linear(state_size, 64)
        # # self.fc2 = nn.Linear(64, 32)
        # # self.dropout = nn.Dropout(p=0.1)

        # Two separate policy heads
        self.policy_action_head = nn.Linear(nn_layer_2, action_size)
        self.policy_draw_head = nn.Linear(nn_layer_2, draw_size)

        # Single value head
        self.value_head = nn.Linear(nn_layer_2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        policy_action_logits = self.policy_action_head(x)
        policy_draw_logits = self.policy_draw_head(x)
        value = self.value_head(x)

        return policy_action_logits, policy_draw_logits, value


def compute_step_reward(state, action, draw_choice, env, step_functions, rule_counter=None):
    # Setup
    step_reward = 0.0
    max_reward = 2.0
    player_hand = state['hands'][state['current_player']]
    is_expedition = action[0] == 'expedition'
    is_number_card = action[1][1] != 'X'
    played_color = action[1][0]
    played_value = int(action[1][1]) if is_number_card else None
    expedition_pile = env.expeditions[state['current_player']][played_color]
    existing_numbers = [int(c[1]) for c in expedition_pile if c[1] != 'X']
    deck_remaining = state['deck_size']
   
    # Bad Move 1: Playing high card when holding lower card
    if 'lover_val_avail' in step_functions:
        same_color_cards = [c for c in player_hand if c[0] == action[1][0] and c[1] != 'X']
        if same_color_cards and action[0] == 'expedition' and is_number_card:
            min_in_hand = min([int(c[1]) for c in same_color_cards])
            if int(action[1][1]) > min_in_hand:
                step_reward -= 1.0
                rule_counter["lower_val_avail"] += 1

    # Bad Move 2: Starting expedition with <6 points in hand
    if 'too_few_pts' in step_functions:
        if is_expedition and len(env.expeditions[state['current_player']][action[1][0]]) == 0:
            color_sum = sum([int(c[1]) for c in player_hand if c[0] == action[1][0] and c[1] != 'X'])
            if color_sum < tgt_pts:
                step_reward -= 0.5
                rule_counter["too_few_pts"] += 1

    # Maybe wrong?
    # Bad Move 3: Starting expedition when opponent blocks reaching 7
    if 'blocked_7' in step_functions:
        opponent = [p for p in env.players if p != state['current_player']][0]
        opp_cards = env.expeditions[opponent].get(action[1][0], [])
        opp_sum = sum([int(c[1]) for c in opp_cards if c[1] != 'X'])
        hand_sum = sum([int(c[1]) for c in player_hand if c[0] == action[1][0] and c[1] != 'X'])
        if is_expedition and len(env.expeditions[state['current_player']][action[1][0]]) == 0:
            if opp_sum + hand_sum < tgt_pts:
                step_reward -= 0.8
                rule_counter["blocked_7"] += 1

    # Bad Move 4: Starting expedition with <=3 cards left
    if 'exp_small_deck' in step_functions:
        if is_expedition and len(env.expeditions[state['current_player']][action[1][0]]) == 0:
            if deck_remaining <= 3:
                step_reward -= 0.5
                rule_counter["exp_small_deck"] += 1

    # Bad Move 5: Discarding to center when expedition is started and playable
    if 'exp_was_live' in step_functions:
        expedition_pile = env.expeditions[state['current_player']][action[1][0]]
        if action[0] == 'center' and expedition_pile:
            top_val = max([int(c[1]) for c in expedition_pile if c[1] != 'X'], default=0)
            card_val = int(action[1][1]) if action[1][1] != 'X' else None
            if card_val is not None and card_val >= top_val:
                step_reward -= 1.5
                rule_counter["exp_was_live"] += 1

    # Good Move: Playing strong expedition (holding >=6 points)
    if 'good_exp' in step_functions:
        if is_expedition:
            color_sum = sum([int(c[1]) for c in player_hand if c[0] == action[1][0] and c[1] != 'X'])
            if color_sum >= tgt_pts-1:
                step_reward += 0.3
                rule_counter["good_exp_1"] += 1
            if color_sum >= tgt_pts:
                step_reward += 1.5
                rule_counter["good_exp"] += 1

    # Penalty 3: Playing RX/BX/GX with no number cards in hand
    if 'bad_X' in step_functions:
        if is_expedition and action[1][1] == 'X':
            same_color_numbers = [c for c in player_hand if c[0] == action[1][0] and c[1] != 'X']
            if not same_color_numbers:
                step_reward -= 2.0
                rule_counter["bad_X"] += 1

    # Penalty 4: Playing R5 while holding R2 or R3
    if 'bad_bigger_val' in step_functions:
        if is_expedition and action[1][1] != 'X':
            played_value = int(action[1][1])
            lower_cards = [int(c[1]) for c in player_hand if c[0] == action[1][0] and c[1] != 'X' and int(c[1]) < played_value]
            if lower_cards:
                step_reward -= 1.5
                rule_counter["bad_bigger_val"] += 1

    # Bonus: Excellent lowest-card play with multiple cards in hand
    if 'good_low_val' in step_functions:
        if is_expedition and is_number_card:
            played_color = action[1][0]
            played_value = int(action[1][1])
            same_color_cards = [int(c[1]) for c in player_hand if c[0] == played_color and c[1] != 'X']
            total_points_in_hand = sum(same_color_cards)
            num_cards_in_hand = len(same_color_cards)
            if num_cards_in_hand >= 2 and played_value <= min(same_color_cards): # DADA and total_points_in_hand >= tgt_pts-1
                step_reward += 1.5
                rule_counter["good_low_val"] += 1

    # Good Draw Move: Drawing card to create >=7 points
    if 'draw_to_tgt' in step_functions:
        if draw_choice in COLORS:
            center_pile = state['center'][draw_choice]
            if center_pile:
                center_card = center_pile[-1]
                if center_card[1] != 'X':
                    center_val = int(center_card[1])
                    color_cards = [int(c[1]) for c in player_hand if c[0] == draw_choice and c[1] != 'X']
                    total_points = sum(color_cards)
                    num_cards = len(color_cards)
                    if num_cards in [1, 2] and (total_points + center_val) >= tgt_pts:
                        step_reward += 1.5
                        rule_counter["draw_to_tgt"] += 1

    # Apply penalty: playing a number card to an empty expedition when holding the multiplier,
    # and when the total known value in hand for this color would make the expedition profitable (7+)
    # Bad Move 7: Playing number card before multiplier when expedition is empty and enough points exist
    if 'had_X' in step_functions:
        if is_expedition and is_number_card:
            played_color = action[1][0]
            expedition_pile = env.expeditions[state['current_player']][played_color]
            
            # Check if expedition is empty
            if not expedition_pile:
                # Get all cards in hand for this color
                color_cards_in_hand = [c for c in player_hand if c[0] == played_color]
                number_points = sum(int(c[1]) for c in color_cards_in_hand if c[1] != 'X')
                has_multiplier = any(c[1] == 'X' for c in color_cards_in_hand)
                
                if has_multiplier and number_points >= tgt_pts and state['deck_size'] >= 5:
                    # Player should have played the multiplier first!
                    step_reward -= 2.0  # penalty can be adjusted
                    rule_counter["had_X"] += 1

    # This is captured above as too_few_pts
    # # Penalty: Starting a new expedition with less than 7 points in hand
    # if is_expedition:
    #     color = action[1][0]
    #     expedition_pile = env.expeditions[state['current_player']][color]
    #     if not expedition_pile:  # Starting new expedition
    #         # color_points = sum(int(c[1]) for c in player_hand if c[0] == color and c[1] in '23456')
    #         color_points = sum(int(c[1]) for c in player_hand if c[0] == color and c[1] != 'X')
    #         if color_points < tgt_pts:
    #             # print(f"Penalty triggered: starting {color} expedition with {color_points} points in hand.")
    #             step_reward -= 1.333
    #             rule_counter["less_7"] += 1

    # OK
    # Bonus: Playing the immediate next card in sequence (no gaps)
    if 'next_value' in step_functions:
        if is_expedition and is_number_card and deck_remaining>=3:
            played_color = action[1][0]
            played_value = int(action[1][1])
        
            expedition_pile = env.expeditions[state['current_player']][played_color]
            existing_numbers = [int(c[1]) for c in expedition_pile if c[1] != 'X']
        
            if existing_numbers:
                top_value = max(existing_numbers)
                if played_value == top_value + 1:
                    step_reward += 0.3
                    rule_counter["next_value"] += 1
                    if random.random() < 1e-8:
                        print(f"Reward playing next value card {action} on {played_color}{top_value}")

    # ❌ Bad Move: Discarding card of color with strong expedition potential and enough deck remaining
    if 'bad_center' in step_functions:
        if action[0] == 'center' and deck_remaining>=5:
            color = action[1][0]
            if not env.expeditions[state['current_player']][color]:  # expedition not started
                color_values = [int(c[1]) for c in player_hand if c[0] == color and c[1] != 'X']
                color_sum = sum(color_values)
                if color_sum >= tgt_pts:
                    step_reward -= 1.25  # Adjust weight if needed
                    rule_counter["bad_center"] += 1
                    if random.random() < 1e-4:
                        print(f"Bad center {action} holding {player_hand}")

    # Reward a smart discard of a value less than the top card on the opp exp pile
    if 'smart_opp_center' in step_functions:
        if action[0] == 'center':
            color = action[1][0]
            card_val = int(action[1][1]) if action[1][1] != 'X' else None
            opp_pile = env.expeditions[opponent].get(color, [])
            opp_vals = [int(c[1]) for c in opp_pile if c[1] != 'X']
            if card_val is not None and any(v > card_val for v in opp_vals):
                step_reward += 0.5
                rule_counter["smart_opp_center"] += 1
                if random.random() < 1e-4:
                    print(f"Smart center play {action} with opp exp {opp_pile}")

    # This is flawed
    # # ✅ Bonus: Closing out expedition – playing highest remaining card in sequence
    # if is_expedition and is_number_card:
    #     color = played_color
    #     value = played_value
    #     # What values of this color are still in hand after this play?
    #     hand_vals = [int(c[1]) for c in player_hand if c[0] == color and c[1] != 'X']
    #     # What’s on the board?
    #     expedition_vals = [int(c[1]) for c in expedition_pile if c[1] != 'X']
        
    #     if hand_vals:
    #         # If this is the max of hand + board, and all other cards are already played
    #         max_val = max(hand_vals + expedition_vals + [value])
    #         if value == max_val and not any(v > value for v in hand_vals):
    #             step_reward += 0.25
    #             # rule_counter["CloseOutExpedition"] += 1
    #             if random.random() < 1e-1:
    #                 print(f"Close out {action} holding {player_hand} on {env.expeditions[state['current_player']][color]}")

    if step_reward>max_reward:
        step_reward=max_reward
        
    return step_reward

def save_config_txt(running, fv, config_dict, step_functions):
    filename = f"config_{running}_{fv}.txt"
    with open(filename, "w") as f:
        f.write(f"Config for version: {fv}\n")
        f.write(f"Generated: {datetime.datetime.now()}\n\n")
        for key, val in config_dict.items():
            f.write(f"{key:<20} = {val}\n")
        f.write("\nStep Reward Functions:\n")
        for fn in step_functions:
            f.write(f"- {fn}\n")
    print(f"Configuration saved to {filename}")

