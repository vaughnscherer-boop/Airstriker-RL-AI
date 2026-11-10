import stable_retro as retro
import numpy as np
import cv2
import torch
from collections import deque
from q_network import QNetwork

env = retro.make("Airstriker-Genesis-v0", render_mode="human")
buttons = env.unwrapped.buttons

actions = [[], ["LEFT"], ["RIGHT"], ["B"], ["LEFT", "B"], ["RIGHT","B"]]
action_table = []

for combo in actions:
    array = np.zeros(12, dtype=np.int8)
    for name in combo:
        position = env.unwrapped.buttons.index(name)
        array[position] = 1
    action_table.append(array)

def step_action(action_number):
    button_array = action_table[action_number]
    total_reward = 0
    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(button_array)
        total_reward += reward
        if terminated or truncated:
            break
    return obs, total_reward, terminated, truncated, info


def test_pattern(pattern_name):
    obs, info = env.reset()
    score = 0

    for i in range(400):
        if pattern_name == "always fire":
            action = 3
            obs, reward, terminated, truncated, info = step_action(action)
            score += reward
            if terminated or truncated:
                break
        elif pattern_name == "alternate":
            if i % 2 == 0:
                action = 3
                obs, reward, terminated, truncated, info = step_action(action)
                score += reward
                if terminated or truncated:
                    break
            else:
                action = 0
                obs, reward, terminated, truncated, info = step_action(action)
                score += reward
                if terminated or truncated:
                    break
        elif pattern_name == "random":
            action = np.random.randint(6)
            obs, reward, terminated, truncated, info = step_action(action)
            score += reward
            if terminated or truncated:
                break
    print(pattern_name, "score:", score, "actions:", i + 1)

test_pattern('always fire')
test_pattern('alternate')
test_pattern('random')




