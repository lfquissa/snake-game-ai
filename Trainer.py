import torch
import torch.nn as nn
import torch.optim as optim


class QTrainer:
    def __init__(self, model, learning_rate, gamma, device):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        torch.manual_seed(0)

    def train_step(self, state_old, action, reward, next_state, game_over):
        state_old = torch.tensor(state_old, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        game_over = torch.tensor(game_over, dtype=torch.bool).to(self.device)

        if len(state_old.shape) == 1:
            state_old = torch.unsqueeze(state_old, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = torch.unsqueeze(game_over, 0)

        # Get predictions for current state
        pred = self.model(state_old)

        # Clone predictions to create target
        target = pred.clone()

        # Calculate Q values for the next state
        with torch.no_grad():
            Q_next = self.model(next_state)

        # Set target values for all samples
        max_Q_next = torch.max(Q_next, dim=1)[0]
        Q_new = reward + self.gamma * max_Q_next * (~game_over)
        target[range(len(action)), action.argmax(dim=1)] = Q_new

        # Perform gradient descent
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()
