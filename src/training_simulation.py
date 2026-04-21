import copy
import math

import numpy as np

from RL_Environment import FIELD_HEIGHT, FIELD_WIDTH, HALF_DURATION


GOAL_Y_MIN = FIELD_HEIGHT / 2 - 110.0
GOAL_Y_MAX = FIELD_HEIGHT / 2 + 110.0
BALL_RADIUS = 12.0


def default_player_specs():
    return (
        {'mass': 73.0, 'radius': 22.0, 'a_max': 100.0, 'v_max': 100.0, 'shot_power_max': 65.0},
        {'mass': 75.0, 'radius': 23.0, 'a_max': 60.0, 'v_max': 95.0, 'shot_power_max': 70.0},
        {'mass': 80.0, 'radius': 24.0, 'a_max': 70.0, 'v_max': 90.0, 'shot_power_max': 50.0},
    )


class TrainingMatchSimulator:
    """Simple deterministic simulator for offline RL training."""

    def __init__(self, total_time=HALF_DURATION):
        self.total_time = total_time
        self.reset()

    def reset(self):
        self.team1_score = 0
        self.team2_score = 0
        self.time_left = self.total_time
        self.half = 1
        self.team1 = self._make_team(side='left')
        self.team2 = self._make_team(side='right')
        self.ball = self._new_ball()
        return self.get_state('left')

    def get_state(self, side='left'):
        if side == 'left':
            return self._build_state(self.team1, self.team2, 'left', self.team1_score, self.team2_score)
        return self._build_state(self.team2, self.team1, 'right', self.team2_score, self.team1_score)

    def step(self, team1_action, team2_action=None):
        team1_action = np.asarray(team1_action, dtype=np.float32)
        team2_action = np.asarray(team2_action if team2_action is not None else np.array([0.0, 0.0]), dtype=np.float32)

        self._move_team(self.team1, team1_action, attacking_right=True)
        self._move_team(self.team2, team2_action, attacking_right=False)
        self._update_ball()
        self._check_goal()

        self.time_left = max(0, self.time_left - 1)
        if self.time_left <= self.total_time / 2:
            self.half = 2

        return self.get_state('left'), self.get_state('right')

    def baseline_action(self, state, player_index=0):
        player = state['our_team'][player_index]
        dx = state['ball']['x'] - player['x']
        dy = state['ball']['y'] - player['y']
        angle = math.atan2(dy, dx)
        force = 0.85 if math.hypot(dx, dy) > 60 else 0.35
        return np.array([(angle % (2 * math.pi)) / (2 * math.pi), force], dtype=np.float32)

    def _new_ball(self):
        return {
            'x': FIELD_WIDTH / 2,
            'y': FIELD_HEIGHT / 2,
            'vx': 0.0,
            'vy': 0.0,
            'alpha': 0.0,
            'v': 0.0,
            'radius': BALL_RADIUS,
        }

    def _build_state(self, our_team, their_team, side, our_score, their_score):
        return {
            'our_team': copy.deepcopy(our_team),
            'their_team': copy.deepcopy(their_team),
            'ball': copy.deepcopy(self.ball),
            'your_side': side,
            'half': self.half,
            'time_left': self.time_left,
            'our_score': our_score,
            'their_score': their_score,
            'total_time': self.total_time,
        }

    def _make_team(self, side):
        x = 240.0 if side == 'left' else FIELD_WIDTH - 240.0
        alpha = 0.0 if side == 'left' else math.pi
        y_positions = (FIELD_HEIGHT * 0.28, FIELD_HEIGHT * 0.5, FIELD_HEIGHT * 0.72)
        team = []
        for y, spec in zip(y_positions, default_player_specs()):
            team.append({
                'x': x,
                'y': y,
                'vx': 0.0,
                'vy': 0.0,
                'alpha': alpha,
                **spec,
            })
        return team

    def _move_team(self, team, controlled_action, attacking_right):
        other_team = self.team2 if team is self.team1 else self.team1
        for index, player in enumerate(team):
            if index == 0:
                action = controlled_action
            else:
                state = self._build_state(team, other_team, 'left' if attacking_right else 'right', 0, 0)
                action = self.baseline_action(state, player_index=index)
            self._apply_action(player, action, attacking_right)

    def _apply_action(self, player, action, attacking_right):
        action = np.clip(np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        player['alpha'] = float(action[0] * 2 * np.pi)
        speed = float(action[1] * player['v_max'])
        drift_x = 0.3 if attacking_right else -0.3
        player['vx'] = math.cos(player['alpha']) * speed * 0.2 + drift_x
        player['vy'] = math.sin(player['alpha']) * speed * 0.2
        player['x'] = float(np.clip(player['x'] + player['vx'], player['radius'], FIELD_WIDTH - player['radius']))
        player['y'] = float(np.clip(player['y'] + player['vy'], player['radius'], FIELD_HEIGHT - player['radius']))

    def _update_ball(self):
        all_players = self.team1 + self.team2
        nearest = min(all_players, key=lambda player: math.hypot(self.ball['x'] - player['x'], self.ball['y'] - player['y']))
        distance = math.hypot(self.ball['x'] - nearest['x'], self.ball['y'] - nearest['y'])
        contact_distance = nearest['radius'] + BALL_RADIUS + 4.0

        if distance <= contact_distance:
            attacking_right = nearest in self.team1
            target_x = FIELD_WIDTH if attacking_right else 0.0
            target_y = FIELD_HEIGHT / 2
            angle = math.atan2(target_y - nearest['y'], target_x - nearest['x'])
            ball_speed = max(10.0, nearest['shot_power_max'] * 0.18)
            self.ball['vx'] = math.cos(angle) * ball_speed
            self.ball['vy'] = math.sin(angle) * ball_speed

        self.ball['x'] += self.ball['vx']
        self.ball['y'] += self.ball['vy']
        self.ball['vx'] *= 0.985
        self.ball['vy'] *= 0.985
        self.ball['alpha'] = math.atan2(self.ball['vy'], self.ball['vx']) if abs(self.ball['vx']) + abs(self.ball['vy']) > 1e-6 else 0.0
        self.ball['v'] = math.hypot(self.ball['vx'], self.ball['vy'])

        if self.ball['y'] <= BALL_RADIUS or self.ball['y'] >= FIELD_HEIGHT - BALL_RADIUS:
            self.ball['vy'] *= -0.9
            self.ball['y'] = float(np.clip(self.ball['y'], BALL_RADIUS, FIELD_HEIGHT - BALL_RADIUS))

        if self.ball['x'] <= BALL_RADIUS and not (GOAL_Y_MIN <= self.ball['y'] <= GOAL_Y_MAX):
            self.ball['vx'] *= -0.9
            self.ball['x'] = BALL_RADIUS
        if self.ball['x'] >= FIELD_WIDTH - BALL_RADIUS and not (GOAL_Y_MIN <= self.ball['y'] <= GOAL_Y_MAX):
            self.ball['vx'] *= -0.9
            self.ball['x'] = FIELD_WIDTH - BALL_RADIUS

    def _check_goal(self):
        if GOAL_Y_MIN <= self.ball['y'] <= GOAL_Y_MAX:
            if self.ball['x'] <= 0:
                self.team1_score += 1
                self._reset_positions()
            elif self.ball['x'] >= FIELD_WIDTH:
                self.team2_score += 1
                self._reset_positions()

    def _reset_positions(self):
        self.team1 = self._make_team(side='left')
        self.team2 = self._make_team(side='right')
        self.ball = self._new_ball()
