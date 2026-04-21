import gymnasium as gym
from gymnasium import spaces
import numpy as np


FIELD_WIDTH = 1366.0
FIELD_HEIGHT = 768.0
HALF_DURATION = 45 * 60


class FootballEnvironment(gym.Env):
    """
    Custom Gymnasium environment for AI Football game.
    Single player control (controls one player on the team).
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self, game_state_provider, player_index=0):
        """
        Args:
            game_state_provider: Function that returns current game state
            player_index: Which player to control (0, 1, or 2)
        """
        super().__init__()
        self.game_state_provider = game_state_provider
        self.player_index = player_index
        
        # Action space: [alpha (direction), force (acceleration)]
        # alpha: 0 to 2*pi (normalized to 0-1)
        # force: 0 to a_max (normalized to 0-1)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        
        # Observation space (28 features)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(28,), 
            dtype=np.float32
        )
        
        self.prev_score_diff = 0
        self.prev_distance_to_ball = 0
        self.step_count = 0

    @staticmethod
    def _player_mass(player):
        return player.get('mass', player.get('weight', 75))

    @staticmethod
    def _ball_velocity(ball):
        if 'vx' in ball or 'vy' in ball:
            return ball.get('vx', 0.0), ball.get('vy', 0.0)

        speed = ball.get('v', 0.0)
        angle = ball.get('alpha', 0.0)
        return speed * np.cos(angle), speed * np.sin(angle)

    @staticmethod
    def _side_flag(your_side):
        if isinstance(your_side, str):
            return 1.0 if your_side.lower() == 'left' else 0.0
        return 1.0 if your_side else 0.0

    def observation_from_state(self, state):
        return self._get_observation(
            state['our_team'],
            state['their_team'],
            state['ball'],
            state['your_side'],
            state['half'],
            state['time_left'],
            state['our_score'],
            state['their_score'],
        )
        
    def _get_observation(self, our_team, their_team, ball, your_side, half, time_left, our_score, their_score):
        """Convert game state to observation vector"""
        player = our_team[self.player_index]
        ball_vx, ball_vy = self._ball_velocity(ball)
        
        # Calculate distance and angle to ball
        dx = ball['x'] - player['x']
        dy = ball['y'] - player['y']
        distance_to_ball = np.sqrt(dx**2 + dy**2)
        angle_to_ball = np.arctan2(dy, dx)
        
        # Normalize angle to 0-1
        normalized_angle = (angle_to_ball + np.pi) / (2 * np.pi)
        
        # Observation vector
        obs = np.array([
            player['x'] / FIELD_WIDTH,
            player['y'] / FIELD_HEIGHT,
            player['alpha'] / (2 * np.pi),
            player['a_max'] / 100.0,
            player['v_max'] / 100.0,
            player['radius'] / 50.0,
            self._player_mass(player) / 100.0,
            player['shot_power_max'] / 100.0,
            their_team[0]['x'] / FIELD_WIDTH,
            their_team[0]['y'] / FIELD_HEIGHT,
            their_team[0]['alpha'] / (2 * np.pi),
            their_team[1]['x'] / FIELD_WIDTH,
            their_team[1]['y'] / FIELD_HEIGHT,
            their_team[1]['alpha'] / (2 * np.pi),
            their_team[2]['x'] / FIELD_WIDTH,
            their_team[2]['y'] / FIELD_HEIGHT,
            their_team[2]['alpha'] / (2 * np.pi),
            ball['x'] / FIELD_WIDTH,
            ball['y'] / FIELD_HEIGHT,
            ball_vx / 50.0,
            ball_vy / 50.0,
            distance_to_ball / 1000.0,
            normalized_angle,
            time_left / HALF_DURATION,
            self._side_flag(your_side),
            half,
            our_score / 10,
            their_score / 10,
        ], dtype=np.float32)
        
        return obs
    
    def reset(self, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_score_diff = 0
        self.prev_distance_to_ball = 0
        
        state = self.game_state_provider()
        if state is None:
            return self.observation_space.sample(), {}
        
        obs = self._get_observation(
            state['our_team'], 
            state['their_team'], 
            state['ball'],
            state['your_side'],
            state['half'],
            state['time_left'],
            state['our_score'],
            state['their_score']
        )
        return obs, {}
    
    def step(self, action):
        """Execute one step with the given action"""
        self.step_count += 1
        
        # Get current game state
        state = self.game_state_provider()
        if state is None:
            return self.observation_space.sample(), 0, False, False, {}
        
        our_team = state['our_team']
        their_team = state['their_team']
        ball = state['ball']
        our_score = state['our_score']
        their_score = state['their_score']
        time_left = state['time_left']
        total_time = state['total_time']
        your_side = state['your_side']
        half = state['half']
        
        # Calculate reward
        reward = self._calculate_reward(
             our_team, their_team, ball, our_score, their_score, time_left, total_time,your_side, half
        )
        
        # Check termination
        terminated = time_left <= 0 
        truncated = False
        
        # Get next observation
        obs = self._get_observation(our_team, their_team, ball, your_side, half, time_left, our_score, their_score)
        
        return obs, reward, terminated, truncated, {}
    
    def _calculate_reward(self, our_team, their_team, ball, our_score, their_score, time_left, total_time,your_side, half):
        """Calculate reward based on game progress"""
        reward = 0.0
        
        player = our_team[self.player_index]
        dx = ball['x'] - player['x']
        dy = ball['y'] - player['y']
        distance_to_ball = np.sqrt(dx**2 + dy**2)
        
        # Reward for scoring
        score_diff = our_score - their_score
        if score_diff > self.prev_score_diff:
            reward += 100.0
        if score_diff < self.prev_score_diff:
            reward -= 50.0
        self.prev_score_diff = score_diff
        
        is_left_side = self._side_flag(your_side) == 1.0
        if is_left_side and ball['x'] > FIELD_WIDTH / 2:
            reward += 10.0
        elif (not is_left_side) and ball['x'] < FIELD_WIDTH / 2:
            reward -= 10.0
        
        # Time-based rewards/penalties
        if time_left <= 0:  # Game ended
            if our_score > their_score:
                reward += 50.0  # Won the game
            elif our_score < their_score:
                reward -= 50.0  # Lost the game
            else:
                reward -= 10.0  # Draw (could be better)
        
        # Small reward for moving toward ball
        if distance_to_ball < self.prev_distance_to_ball:
            reward += 0.5
        else:
            reward -= 0.1
        self.prev_distance_to_ball = distance_to_ball
        
        # Penalty for being far from ball
        if distance_to_ball > 300:
            reward -= 0.2
        
        # Time pressure - reward urgency near end of game
        time_ratio = time_left / max(total_time, 1)
        if time_ratio < 0.2:  # Last 20% of game
            if our_score <= their_score:  # Behind or tied
                reward -= 0.5  # Urgency penalty
            else:  # Winning
                reward += 0.2  # Maintain lead bonus
        
        # Small reward for staying alive and active
        reward += 0.01
        
        # reward for moving
       # speed = np.sqrt(player.get('vx', 0)**2 + player.get('vy', 0)**2)
      #  reward += speed * 0.1
            
            
       
        
        
        return float(reward)
    
    def set_game_state(self, our_team, their_team, ball, your_side, half, time_left, our_score, their_score):
        """Update game state for the environment"""
        self.current_state = {
            'our_team': our_team,
            'their_team': their_team,
            'ball': ball,
            'your_side': your_side,
            'half': half,
            'time_left': time_left,
            'our_score': our_score,
            'their_score': their_score,
            'total_time': 45 * 60
        }
    
    def render(self):
        """Render is handled by pygame in the main game loop"""
        pass
