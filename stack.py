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

env = retro.make("Airstriker-Genesis-v0", render_mode=None)
obs, info = env.reset()

first_frame = preprocess(obs)
frames = deque(maxlen=4)

for i in range(4):
    frames.append(obs)

no_buttons = np.zeros(12, dtype=np.int8)

for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(no_buttons)
    frames.append(first_frame)

state = np.stack(frames)

state_tensor = torch.from_numpy(state).float() / 255

state_tensor = state_tensor.unsqueeze(0)

model = QNetwork()
with torch.no_grad():
    q_values = model(state_tensor)

action = q_values.argmax(dim=1).item()

env.close()