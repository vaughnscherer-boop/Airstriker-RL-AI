import stable_retro as retro
import numpy as np
import cv2
import torch
from collections import deque
from q_network import QNetwork

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return small

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

def make_state(frames):
    state = np.stack(frames)
    state_tensor = torch.from_numpy(state).float() / 255
    state_tensor = state_tensor.unsqueeze(0)
    return state_tensor

def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.randint(6)
    else:
        with torch.no_grad():
            q_values = model(state)

        action = q_values.argmax(dim=1).item()
    return action

model = QNetwork()
epsilon = 0.1

obs, info = env.reset()

first_frame = preprocess(obs)
frames = deque(maxlen=4)

for i in range(4):
    frames.append(first_frame)

score = 0
steps = 0

while True:
    state = make_state(frames)
    action = choose_action(state, epsilon)
    obs, reward, terminated, truncated, info = step_action(action)
    result = preprocess(obs)
    frames.append(result)
    score += reward
    steps += 1
    print(steps, action, reward)
    if terminated or truncated:
        break

print(score)
print(steps)

try:
    env.close()
except AttributeError:
    pass