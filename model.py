import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
       
class CNN_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO currently hardcoded for a 32x24 game.
        # 'out channels' is the number of output features - i.e. hidden units.
        self.conv1 = nn.Conv2d(in_channels = 4, out_channels=16, stride=1, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels=32, stride=1, kernel_size=2)
        
        # We have to flatten the image for input to the FC layer.
        # Each of the 12 hidden units in conv2 is of size w x h.
        # Thus input to FC1 is hidden units * w x h.
        # w x h has to be worked out from conv1, as downsampling may happen.
        self.fc1 = nn.Linear(32*27*19, 256)
        # One output for each action: left, up, right.
        self.fc2 = nn.Linear(256, 3)
      
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Have to move from 2D representation to 1D for linear layer.
        x = x.view(-1, 32*27*19)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 3:
            # A single training example. For 2D convolution, Pytorch expects shape
            # [batch_size, in_channels, in_height, in_width]. Currently of shape
            # [in_height, in_width]. Unsqueeze adds the extra dimensions of depth 1.
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0)))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        
def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features



