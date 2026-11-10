import stable_retro as retro
import numpy as np
import cv2
import torch
from collections import deque
from q_network import QNetwork
from replay_buffer import ReplayBuffer
import torch.nn as nn

TOTAL_STEPS = 2000000
BUFFER_SIZE = 50_000
BATCH_SIZE = 64
LEARNING_STARTS = 10_000
TRAIN_EVERY = 8
TARGET_UPDATE = 10_000
EPSILON_DECAY_STEPS = 250_000
EPSILON_MIN = 0.05
MAX_GAME_STEPS = 7500

device = torch.device("mps")
buffer = ReplayBuffer(BUFFER_SIZE)

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
            q_values = model(state.to(device))

        action = q_values.argmax(dim=1).item()
    return action

def get_epsilon(total_steps):
    return EPSILON_MIN

model = QNetwork().to(device)
model.load_state_dict(torch.load("airstriker_dqn_run3_best.pt", map_location=device))
target_model = QNetwork().to(device)
target_model.load_state_dict(model.state_dict())

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.SmoothL1Loss()
gamma = 0.99

epsilon = .05
total_steps = 0
game = 0
all_scores = []
losses = []
best_average = 0

while total_steps < TOTAL_STEPS:
    game += 1
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
        lives = info["lives"]
        buffer.add(state_array, action, stored_reward, next_state_array, terminated)
        score += reward
        steps += 1
        total_steps += 1
        epsilon = get_epsilon(total_steps)

        if len(buffer) >= LEARNING_STARTS and total_steps % TRAIN_EVERY == 0:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            q_values = model(states)
            q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_model(next_states).max(dim=1).values
                targets = rewards + gamma * next_q * (1 - dones)
            loss = loss_fn(q_taken, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if total_steps % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())
        
        if terminated or truncated:
            break

        if steps >= MAX_GAME_STEPS:
            break

    all_scores.append(score)
    average = np.mean(all_scores[-20:])
    if game >= 20 and average > best_average:                 # NEW
        best_average = average                                 # NEW
        torch.save(model.state_dict(), "airstriker_dqn_run4_best.pt")  # NEW
        print(f"New best average {best_average:.1f}, saved")

    print(f'game: {game}| score {score}| average {average:.1f}| epsilon {epsilon:.3f}| total_steps {total_steps}')
    
    if game % 25 == 0:
        torch.save(model.state_dict(), "airstriker_dqn_run4.pt")
        print("Saved model")

torch.save(model.state_dict(), "airstriker_dqn_run4.pt")
print("Training finished")

try:
    env.close()
except AttributeError:
    pass

