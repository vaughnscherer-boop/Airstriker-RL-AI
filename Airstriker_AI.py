import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, num_actions=6):
        super().__init__()
        # Layers with weights are created here.
        # Conv2d(input channels, output channels, filter size, step size)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)  # keep the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # no ReLU: Q-values can be negative
        return x


model = QNetwork()
fake_frames = torch.rand(32, 4, 84, 84)  # 32 states, 4 frames each, 84x84 pixels
predictions = model(fake_frames)



targets = torch.rand(32, 6)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.SmoothL1Loss()
losses = []

for epoch in range(200):
    predictions = model(fake_frames)
    loss = loss_fn(predictions, targets)
    losses.append(loss.item())
    if epoch % 20 == 0:
        print(f'Epoch: {epoch} and loss: {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



    