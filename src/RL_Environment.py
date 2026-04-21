import gymnasium as gym
from gymnasium import spaces
import numpy as np


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
        
        # Observation space (19 features)
        # Player: x, y, vx, vy, alpha, a_max, v_max, radius, mass, shot_power
        # Ball: x, y, vx, vy
        # Game: score_diff, time_left_ratio, distance_to_ball, angle_to_ball
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(19,), 
            dtype=np.float32
        )
        
        self.prev_score_diff = 0
        self.prev_distance_to_ball = 0
        self.step_count = 0
        
    def _get_observation(self, our_team, their_team, ball, time_left, total_time):
        """Convert game state to observation vector"""
        player = our_team[self.player_index]
        
        # Calculate distance and angle to ball
        dx = ball['x'] - player['x']
        dy = ball['y'] - player['y']
        distance_to_ball = np.sqrt(dx**2 + dy**2)
        angle_to_ball = np.arctan2(dy, dx)
        
        # Normalize angle to 0-1
        normalized_angle = (angle_to_ball + np.pi) / (2 * np.pi)
        
        # Observation vector
        obs = np.array([
            # Player state  features)
            player['x'] / 1366.0,  # Normalize by screen width
            player['y'] / 768.0,   # Normalize by screen height
            player['alpha'] / (2 * np.pi),
            player['a_max', 100] / 100.0,
            player['v_max', 100] / 100.0,
            player['radius', 20] / 50.0,
            player['weight', 75] / 100.0,
            player['shot_power_max'] / 100.0,
            
            # Ball state  features)
            ball['x'] / 1366.0,
            ball['y'] / 768.0,
            ball.get('vx', 0) / 50.0,
            ball.get('vy', 0) / 50.0,
            
            # Game state features)
            distance_to_ball / 1000.0,
            normalized_angle,
            time_left / (45 * 60),  # Normalize by half duration
            
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
            state['time_left'],
            state['total_time']
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
        
        # Calculate reward
        reward = self._calculate_reward(
            our_team, their_team, ball, 
            our_score, their_score,
            time_left, total_time
        )
        
        # Check termination
        terminated = time_left <= 0 or len(our_team) == 0
        truncated = False
        
        # Get next observation
        obs = self._get_observation(our_team, their_team, ball, time_left, total_time)
        
        return obs, reward, terminated, truncated, {}
    
    def _calculate_reward(self, our_team, their_team, ball, our_score, their_score, time_left, total_time):
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
        time_ratio = time_left / total_time
        if time_ratio < 0.2:  # Last 20% of game
            if our_score <= their_score:  # Behind or tied
                reward -= 0.5  # Urgency penalty
            else:  # Winning
                reward += 0.2  # Maintain lead bonus
        
        # Small reward for staying alive and active
        reward += 0.01
        
        # reward for moving
        speed = np.sqrt(player.get('vx', 0)**2 + player.get('vy', 0)**2)
        reward += speed * 0.1
            
            
       
        
        
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
