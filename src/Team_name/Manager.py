import numpy as np
import os
from stable_baselines3 import PPO

# ============================================================================
# REINFORCEMENT LEARNING CONFIGURATION
# ============================================================================
rl_model = None
use_rl_for_player = [True, False, False]  # Use RL for player 0 only
rl_ready = False

def load_rl_model():
    """Load trained RL model"""
    global rl_model, rl_ready
    
    model_path = os.path.join(os.path.dirname(__file__), '../../models/rl_football_agent')
    
    try:
        if os.path.exists(model_path + '.zip'):
            rl_model = PPO.load(model_path)
            rl_ready = True
            print("✓ RL Model loaded successfully!")
            return True
        else:
            rl_ready = False
            print("⚠ RL model not found. Training script not run yet.")
            return False
    except Exception as e:
        rl_ready = False
        print(f"⚠ Failed to load RL model: {e}")
        return False

def get_rl_observation(player, ball, other_players, score_diff, time_left):
    """Convert game state to RL observation vector (19 features)"""
    
    # Calculate distance and angle to ball
    dx = ball['x'] - player['x']
    dy = ball['y'] - player['y']
    distance_to_ball = np.sqrt(dx**2 + dy**2)
    angle_to_ball = np.arctan2(dy, dx)
    
    # Strength comparison - use shot_power_max or shot_power
    player_shot = player.get('shot_power_max', player.get('shot_power', 50))
    other_shot = other_players[0].get('shot_power_max', other_players[0].get('shot_power', 50))
    strength_diff = (player_shot - other_shot) / 100.0
    
    obs = np.array([
        # Player state (10 features)
        player['x'] / 1366.0,
        player['y'] / 768.0,
        player.get('vx', 0) / 100.0,
        player.get('vy', 0) / 100.0,
        player['alpha'] / (2 * np.pi),
        player.get('a_max', 100) / 100.0,
        player.get('v_max', 100) / 100.0,
        player.get('radius', 20) / 50.0,
        player.get('weight', 75) / 100.0,
        player_shot / 100.0,
        
        # Ball state (4 features)
        ball['x'] / 1366.0,
        ball['y'] / 768.0,
        ball.get('vx', 0) / 50.0,
        ball.get('vy', 0) / 50.0,
        
        # Game state (5 features)
        distance_to_ball / 1000.0,
        (angle_to_ball + np.pi) / (2 * np.pi),
        time_left / (45 * 60),
        strength_diff,
        0.0,
    ], dtype=np.float32)
    
    return obs

def action_to_decision(action, player):
    """Convert RL action (normalized 0-1) to decision dict"""
    # action[0]: alpha (direction) - 0 to 1 maps to 0 to 2*pi
    # action[1]: force - 0 to 1 maps to 0 to a_max
    
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
    # Choose a funny name for each player and your team
    # Use names written only in cyrillic
    # Make sure that the name is less than 11 characters
    # Don't use profanity!!!
def team_properties():
    properties = dict()
    player_names = ["Пандевалдо", "Панчевалдо", "Елмасалдо"]
    properties['team_name'] = "Мак Челзи"
    properties['player_names'] = player_names
    properties['image_name'] = 'Red.png' # use image resolution 153x153
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
        
        if use_rl_for_player[i] and rl_ready:
            # Use RL agent for this player
            try:
                obs = get_rl_observation(player, ball, their_team, score_diff, time_left)
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
            # Simple baseline strategy for other players
            manager_decision[i]['alpha'] = player['alpha']
            manager_decision[i]['force'] = 0.7 * player['a_max']
            manager_decision[i]['shot_request'] = False
            manager_decision[i]['shot_power'] = 50
    
    return manager_decision
    # print(our_score, their_score)
    # print(our_team[0]['weight'], our_team[0]['radius'], our_team[0]['max_acceleration'], our_team[0]['max_speed'], our_team[0]['shot_power'])
    # print(their_team[0]['weight'], their_team[0]['radius'], their_team[0]['max_acceleration'], their_team[0]['max_speed'], their_team[0]['shot_power'])
    
    return manager_decision
