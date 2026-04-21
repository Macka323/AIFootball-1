
import numpy as np
import os
from stable_baselines3 import PPO

# ============================================================================
# SELF-PLAY RL CONFIGURATION
# ============================================================================
rl_model = None
rl_ready = False


def _player_mass(player):
    return player.get('mass', player.get('weight', 75))


def _ball_velocity(ball):
    if 'vx' in ball or 'vy' in ball:
        return ball.get('vx', 0.0), ball.get('vy', 0.0)

    speed = ball.get('v', 0.0)
    angle = ball.get('alpha', 0.0)
    return speed * np.cos(angle), speed * np.sin(angle)


def _side_flag(your_side):
    if isinstance(your_side, str):
        return 1.0 if your_side.lower() == 'left' else 0.0
    return 1.0 if your_side else 0.0

def load_rl_model():
    """Load trained self-play RL model"""
    global rl_model, rl_ready
    
    model_path = os.path.join(os.path.dirname(__file__), '../../models/selfplay_rl_football_agent')
    
    try:
        if os.path.exists(model_path + '.zip'):
            rl_model = PPO.load(model_path)
            rl_ready = True
            print("✓ Self-play RL Model loaded successfully!")
            return True
        else:
            rl_ready = False
            print("⚠ Self-play RL model not found. Train it first with: python train_selfplay_rl_agent.py")
            return False
    except Exception as e:
        rl_ready = False
        print(f"⚠ Failed to load RL model: {e}")
        return False

def get_rl_observation(player, their_team, ball, your_side, half, time_left, our_score, their_score):
    """Convert game state to the 28-feature RL observation vector."""
    ball_vx, ball_vy = _ball_velocity(ball)

    # Calculate distance and angle to ball
    dx = ball['x'] - player['x']
    dy = ball['y'] - player['y']
    distance_to_ball = np.sqrt(dx**2 + dy**2)
    angle_to_ball = np.arctan2(dy, dx)
    
    
    obs = np.array([
        # Player state
        player['x'] / 1366.0,
        player['y'] / 768.0,
        player['alpha'] / (2 * np.pi),
        player['a_max']  / 100.0,
        player['v_max']  / 100.0,
        player['radius'] / 50.0,
        _player_mass(player) / 100.0,
        player['shot_power_max'] / 100.0,
        
        their_team[0]['x'] / 1366.0,  # Normalize by screen width
        their_team[0]['y'] / 768.0,   # Normalize by screen height
        their_team[0]['alpha'] / (2 * np.pi),
        their_team[1]['x'] / 1366.0,  # Normalize by screen width
        their_team[1]['y'] / 768.0,   # Normalize by screen height
        their_team[1]['alpha'] / (2 * np.pi),
        their_team[2]['x'] / 1366.0,  # Normalize by screen width
        their_team[2]['y'] / 768.0,   # Normalize by screen height
        their_team[2]['alpha'] / (2 * np.pi),
        
        # Ball state
        ball['x'] / 1366.0,
        ball['y'] / 768.0,
        ball_vx / 50.0,
        ball_vy / 50.0,
        
        # Game state
        distance_to_ball / 1000.0,
        (angle_to_ball + np.pi) / (2 * np.pi),
        time_left / (45 * 60),
        _side_flag(your_side),
        half,
        our_score/10,
        their_score/10
    ], dtype=np.float32)
    
    return obs

def action_to_decision(action, player):
    """Convert RL action (normalized 0-1) to decision dict"""
    try:
        # Ensure action is numpy array and handle NaN/inf
        action = np.array(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.5, posinf=1.0, neginf=0.0)
        action = np.clip(action, 0.0, 1.0)
        
        alpha = float(action[0] * 2 * np.pi)
        force = float(action[1] * player['a_max'])
        
        # Validate outputs
        if not np.isfinite(alpha):
            alpha = player['alpha']
        if not np.isfinite(force):
            force = 0.5 * player['a_max']
        
        decision = {
            'alpha': alpha,
            'force': force,
            'shot_request': False,
            'shot_power': 50,
        }
        
        return decision
    except Exception as e:
        print(f"⚠ Action conversion failed: {e}")
        return {
            'alpha': player['alpha'],
            'force': 0.5 * player['a_max'],
            'shot_request': False,
            'shot_power': 50,
        }

# Choose names for your players and team
def team_properties():
    properties = dict()
    player_names = ["Пандевалдо", "Панчевалдо", "Елмасалдо"]
    properties['team_name'] = "Мак Челзи"
    properties['player_names'] = player_names
    properties['image_name'] = 'Red.png'
    properties['weight_points'] = (9, 10, 15)
    properties['radius_points'] = (5, 10, 20)
    properties['max_acceleration_points'] = (40, 10, 15)
    properties['max_speed_points'] = (40, 10, 25)
    properties['shot_power_points'] = (18, 20, 13)
    return properties

# This function gathers game information and controls each one of your three players
def decision(our_team, their_team, ball, your_side, half, time_left, our_score, their_score):
    
    # Load RL model on first call
    global rl_model, rl_ready
    if rl_model is None and not rl_ready:
        load_rl_model()
    
    manager_decision = [dict(), dict(), dict()]
    score_diff = our_score - their_score
    
    for i in range(3):
        player = our_team[i]
        
        if rl_ready:
            # Use RL agent for this player
            try:
                obs = get_rl_observation(our_team[i], their_team, ball, your_side, half, time_left, our_score, their_score)
                action, _ = rl_model.predict(obs, deterministic=True)
                manager_decision[i] = action_to_decision(action, player)
            except Exception as e:
                # Fallback to simple strategy if RL fails
                print(f"⚠ RL prediction failed for player {i}: {e}")
                manager_decision[i]['alpha'] = player['alpha']
                manager_decision[i]['force'] = 0.5 * player['a_max']
                manager_decision[i]['shot_request'] = False
                manager_decision[i]['shot_power'] = 50
        else:
            # Simple baseline strategy
            manager_decision[i]['alpha'] = player['alpha']
            manager_decision[i]['force'] = 0.7 * player['a_max']
            manager_decision[i]['shot_request'] = False
            manager_decision[i]['shot_power'] = 50
    
    return manager_decision
