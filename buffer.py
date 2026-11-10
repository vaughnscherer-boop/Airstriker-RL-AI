import stable_retro as retro
import numpy as np
import cv2
import torch
from collections import deque
from q_network import QNetwork
from replay_buffer import ReplayBuffer

buffer = ReplayBuffer(50000)

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return small

env = retro.make("Airstriker-Genesis-v0", render_mode=None)
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
lives = 3

while True:
    state_array = np.stack(frames)
    state = make_state(frames)
    action = choose_action(state, epsilon)
    obs, reward, terminated, truncated, info = step_action(action)
    result = preprocess(obs)
    frames.append(result)
    next_state_array = np.stack(frames)
    stored_reward = reward / 20
    if info["lives"] < lives:
        stored_reward -= 1
        print("Life lost at step", steps, "stored reward:", stored_reward)
    lives = info["lives"]
    buffer.add(state_array, action, stored_reward, next_state_array, terminated)
    score += reward
    steps += 1
    if terminated or truncated:
        break

print(len(buffer))
print(score)
print(steps)
states, actions, rewards, next_states, dones = buffer.sample(32)
print(states.shape, states.dtype, states.min(), states.max())
print(actions.shape, actions.dtype)
print(rewards.shape, rewards.dtype)
print(dones.shape, dones.dtype)

try:
    env.close()
except AttributeError:
    pass