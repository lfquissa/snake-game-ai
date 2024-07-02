# Snake Game with Reinforcement Learning

This project implements a classic Snake Game using PyGame and trains an AI agent to play the game using Deep Q-Learning (DQN) with PyTorch.

## Project Overview

The Snake Game is implemented with PyGame, providing a graphical interface for the game. An AI agent is then created to learn and play the game autonomously using reinforcement learning techniques, specifically the DQN algorithm.

### Key Components

1. **Snake Game**: Implemented using PyGame, providing the game environment.
2. **AI Agent**: Uses Deep Q-Learning to make decisions and improve gameplay.
3. **Neural Network**: Implemented with PyTorch for Q-value approximation.
4. **Training Process**: The agent learns through repeated gameplay and experience.

## Features

- Classic Snake Game implementation
- AI agent that learns to play the game
- DQN algorithm for reinforcement learning
- PyTorch implementation of the neural network
- Customizable hyperparameters for training
- Evaluation mode to test the trained agent
- Performance tracking and visualization

## Requirements

- Python 3.x
- PyGame
- PyTorch
- Matplotlib
- NumPy

## Project Structure

- `Main.py`: Entry point for training and evaluation
- `SnakeGame.py`: Implementation of the Snake Game
- `Agent.py`: AI agent implementation with DQN
- `Model.py`: Neural network model definition
- `Trainer.py`: Training logic for the agent
- `Constants.py`: Game and training constants
- `Action.py`, `Directions.py`, `Point.py`: Utility classes

## Usage

1. Train the agent:
   ```
   python Main.py
   ```

2. Evaluate the trained agent:
   ```
   python Main.py --eval-mode
   ```

## Customization

You can adjust various hyperparameters in `Constants.py` to experiment with different training configurations.

## Output

The training process will generate:
- Checkpoint files during training
- A final model file after training
- Performance plots (score, mean score, and loss)

## Notes

- The agent uses a combination of exploration and exploitation strategies during training.
- Performance may vary depending on the chosen hyperparameters and training duration.

## Future Improvements

- Implement more advanced reinforcement learning algorithms
- Add more complex game features to increase difficulty
- Optimize the neural network architecture for better performance
