import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import sys
import os
from collections import deque
from game import SnakeGameAI, Direction, Point
# from model import LinearQNet#, QTrainer, CNNModel
from helper import plot
import time
from os import path

# sbatch cifar10_standard_single_gpu_tutorial.sh

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
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
        state = state.to(device)
        next_state = next_state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        # (n, x)
        # print("st", state)
        # print("sh", state.shape)
        if len(state.shape) == 1:
            # (1, x)
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

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


class NewAgent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 2000
        # randomness, in this implementation it is actually how many episodes you want the agent to be able to act
        # randomly. After 'epsilon' episodes the agent will not be able to act randomly at all. Until then, it will
        # become gradually less and less random.
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        # self.model = CNNModel()
        self.model = LinearQNet(111, 512, 3).to(device)
        # self.model.load_state_dict(torch.load("/Users/azamkhan/github/mlp-cw3/mlp-cw3-evan_changes/model/model.pth"))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        grid = game.grid

        apple_grid = game.apple_grid
        body_grid = game.body_grid
        head_grid = game.head_grid

        # print("a", apple_grid)
        # print("b", body_grid)
        # print("h", head_grid)

        grid = np.array([apple_grid, body_grid, head_grid])
        # print(state)
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        state = np.array(state, dtype=int)

        # Expand dim for number of channels, since it is "black and white" the number of channels is 1
        # state = np.expand_dims(state, 0)
        # Un-squeeze for batch size

        fstate = np.concatenate((game.grid.flatten(), state), axis=None)
        # print (fstate)
        return fstate

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        # state = torch.tensor(state, dtype=torch.float)
        # next_state = torch.tensor(next_state, dtype=torch.float)
        # action = torch.tensor(action, dtype=torch.long)
        # reward = torch.tensor(reward, dtype=torch.float)

        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):

        # random moves: tradeoff exploration / exploitation
        epsilon = self.epsilon - self.n_games
        # print(epsilon)
        final_move = [0, 0, 0]
        if random.randint(0, self.epsilon) < epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            state0 = state0.to(device)
            prediction = self.model(state0)

            move = torch.argmax(prediction).item()
            final_move[move] = 1
            # print(prediction)
        return final_move


# class Agent:
#
#     def __init__(self):
#         self.n_games = 0
#         self.epsilon = 0 # randomness
#         self.gamma = 0.9 # discount rate
#         self.memory = deque(maxlen=MAX_MEMORY) # popleft()
#         self.model = LinearQNet(11, 256, 3)
#         self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
#
#     def get_state(self, game):
#         head = game.snake[0]
#         point_l = Point(head.x - 20, head.y)
#         point_r = Point(head.x + 20, head.y)
#         point_u = Point(head.x, head.y - 20)
#         point_d = Point(head.x, head.y + 20)
#
#         dir_l = game.direction == Direction.LEFT
#         dir_r = game.direction == Direction.RIGHT
#         dir_u = game.direction == Direction.UP
#         dir_d = game.direction == Direction.DOWN
#
#         state = [
#             # Danger straight
#             (dir_r and game.is_collision(point_r)) or
#             (dir_l and game.is_collision(point_l)) or
#             (dir_u and game.is_collision(point_u)) or
#             (dir_d and game.is_collision(point_d)),
#
#             # Danger right
#             (dir_u and game.is_collision(point_r)) or
#             (dir_d and game.is_collision(point_l)) or
#             (dir_l and game.is_collision(point_u)) or
#             (dir_r and game.is_collision(point_d)),
#
#             # Danger left
#             (dir_d and game.is_collision(point_r)) or
#             (dir_u and game.is_collision(point_l)) or
#             (dir_r and game.is_collision(point_u)) or
#             (dir_l and game.is_collision(point_d)),
#
#             # Move direction
#             dir_l,
#             dir_r,
#             dir_u,
#             dir_d,
#
#             # Food location
#             game.food.x < game.head.x,  # food left
#             game.food.x > game.head.x,  # food right
#             game.food.y < game.head.y,  # food up
#             game.food.y > game.head.y  # food down
#             ]
#
#         return np.array(state, dtype=int)
#
#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
#
#     def train_long_memory(self):
#         if len(self.memory) > BATCH_SIZE:
#             mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
#         else:
#             mini_sample = self.memory
#         states, actions, rewards, next_states, dones = zip(*mini_sample)
#         self.trainer.train_step(states, actions, rewards, next_states, dones)
#         # for state, action, reward, nexrt_state, done in mini_sample:
#         #     self.trainer.train_step(state, action, reward, next_state, done)
#
#     def train_short_memory(self, state, action, reward, next_state, done):
#         self.trainer.train_step(state, action, reward, next_state, done)
#
#     def get_action(self, state):
#         # random moves: tradeoff exploration / exploitation
#         self.epsilon = 80 - self.n_games
#         final_move = [0,0,0]
#         if random.randint(0, 200) < self.epsilon:
#             move = random.randint(0, 2)
#             final_move[move] = 1
#         else:
#             state0 = torch.tensor(state, dtype=torch.float)
#             prediction = self.model(state0)
#             move = torch.argmax(prediction).item()
#             final_move[move] = 1
#
#         return final_move


def train():
    plot_scores = []
    plot_mean_scores = []

    filename = time.strftime("%Y_%m_%d-%H_%M_%S.txt")

    f = open(path.join('./model', filename), "a")
    f.write("{}, {}, {}".format("Game number", "Score", "Record"))
    f.close()

    total_score = 0
    record = 0
    agent = NewAgent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move

        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            f = open(path.join('./model', filename), "a")
            f.write("{}, {}, {}\n".format(agent.n_games, score, record))
            f.close()

            # plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
