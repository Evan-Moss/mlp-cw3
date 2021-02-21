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
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=6, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels=12, kernel_size=(3,3))
        
        # TODO: 200 is arbitrary here - try to find better values.
        # In size is based on image size and proceding downsampling from kernels.
        # This is currently hardcoded for image size 32x24.
        # 32x24 Original size. Kernel size 3x3. Stride 1.
        self.fc1 = nn.Linear(12*28*20, 50)
        # One output for each action: left, up, right.
        self.fc2 = nn.Linear(50, 3)
      
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Have to move from 2D representation to 1D for linear layer.
        x = x.view(-1, 12*28*20)
        
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

        if len(state.shape) == 2:
            # A single training example. For 2D convolution, Pytorch expects shape
            # [batch_size, in_channels, in_height, in_width]. Currently of shape
            # [in_height, in_width]. Unsqueeze adds the extra dimensions of depth 1.
            state = state.unsqueeze(0).unsqueeze(0)
            next_state = next_state.unsqueeze(0).unsqueeze(0)
            action = action.unsqueeze(0).unsqueeze(0)
            reward = reward.unsqueeze(0).unsqueeze(0)
            done = (done, )
        else:
            # Already have batches. Need to add the number of channels (1) to the shape.
            state = state.unsqueeze(1)
            next_state = next_state.unsqueeze(1)
            #action = action.unsqueeze(1)
            #reward = reward.unsqueeze(1)

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



