import stable_retro as retro
import numpy as np
import cv2

env = retro.make("Airstriker-Genesis-v0", render_mode=None)
obs, info = env.reset()

no_buttons = np.zeros(12, dtype=np.int8)

for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(no_buttons)

print(obs.shape, obs.dtype)
print("Score:", info["score"], "Lives:", info["lives"])

gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

small = cv2.resize(gray, (84, 84), cv2.INTER_AREA)
print(small.shape)

cv2.imwrite("frame_gray_84.png", small)
env.close()