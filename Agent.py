# Agent.py
import datetime
import json
import os
import random
import shutil
from collections import deque

import torch

import Constants
from Action import Action
from Model import Linear_QNet
from Trainer import QTrainer


class Agent:
    def __init__(self, eval_mode=False, hyperparameters=None):
        self.n_games = 0
        self.eval_mode = eval_mode
        self.epsilon = 0

        # Initialize hyperparameters with default values from Constants
        self.hyperparameters = hyperparameters or {
            "gamma": Constants.GAMMA,
            "max_memory": Constants.MAX_MEMORY,
            "lr": Constants.LR,
            "batch_size": Constants.BATCH_SIZE,
            "training_episodes": Constants.TRAINING_EPISODES,
            "epsilon": Constants.EPSILON,
            "eps_upper": Constants.EPS_UPPER,
        }

        self.eps_upper = self.hyperparameters["eps_upper"]
        self.gamma = self.hyperparameters["gamma"]
        self.memory = deque(maxlen=self.hyperparameters["max_memory"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Linear_QNet().to(self.device)
        self.trainer = QTrainer(
            self.model, self.hyperparameters["lr"], self.gamma, self.device
        )

        # Create a unique directory for checkpoints based on hyperparameters
        self.checkpoint_dir = self.create_unique_checkpoint_dir()

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Save hyperparameters to a JSON file in the checkpoint directory
        self.save_hyperparameters()

    def create_unique_checkpoint_dir(self):
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%Hh%Mm%Ss")
        hyperparam_str = "-".join(
            [f"{key}={value}" for key, value in self.hyperparameters.items()]
        )
        checkpoint_dir = f"./checkpoints/{timestamp}-{hyperparam_str}"
        return checkpoint_dir

    def save_hyperparameters(self):
        hyperparameters_path = os.path.join(self.checkpoint_dir, "hyperparameters.json")
        with open(hyperparameters_path, "w") as f:
            json.dump(self.hyperparameters, f, indent=4)

    def add_games_count(self):
        self.n_games += 1

    def save_checkpoint(self, episode, total_score, record, metrics):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode}.pth")
        torch.save(
            {
                "episode": episode,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
                "total_score": total_score,
                "record": record,
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return (
            checkpoint["episode"],
            checkpoint["total_score"],
            checkpoint["record"],
            checkpoint["metrics"],
        )

    def load_model(self, file_name="./model/model.pth"):
        self.model.load_model(file_name)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > self.hyperparameters["batch_size"]:
            mini_sample = random.sample(self.memory, self.hyperparameters["batch_size"])
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        return self.trainer.train_step(
            states, actions, rewards, next_states, game_overs
        )

    def get_action(self, game_state):
        final_move = [0, 0, 0]

        if self.eval_mode:
            with torch.no_grad():
                state = torch.tensor(game_state, dtype=torch.float).to(self.device)
                prediction = self.model(state)
                move = torch.argmax(prediction).item()
                final_move[move] = 1
                index = final_move.index(1)
                if index == 0:
                    return Action.STRAIGHT
                if index == 1:
                    return Action.RIGHT
                if index == 2:
                    return Action.LEFT

        self.epsilon = Constants.EPSILON - self.n_games

        if random.randint(0, self.eps_upper) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state = torch.tensor(game_state, dtype=torch.float).to(self.device)
            prediction = self.model(state)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        index = final_move.index(1)
        if index == 0:
            return Action.STRAIGHT
        if index == 1:
            return Action.RIGHT
        if index == 2:
            return Action.LEFT

    def finalize_training(self, final_metrics):
        # Create final directory for model and metrics
        final_dir = self.checkpoint_dir.replace("checkpoints", "final_models")
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)

        # Save final model
        final_model_path = os.path.join(final_dir, "final_model.pth")
        torch.save(self.model.state_dict(), final_model_path)

        # Save final metrics
        final_metrics_path = os.path.join(final_dir, "final_metrics.json")
        with open(final_metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=4)

        # Save hyperparameters again for easy reference
        final_hyperparameters_path = os.path.join(final_dir, "hyperparameters.json")
        with open(final_hyperparameters_path, "w") as f:
            json.dump(self.hyperparameters, f, indent=4)

        # Remove checkpoint directory
        shutil.rmtree(self.checkpoint_dir)

    def get_final_model_dir(self):
        return self.checkpoint_dir.replace("checkpoints", "final_models")
