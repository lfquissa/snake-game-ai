import os
import time
import argparse
from collections import deque

import matplotlib.pyplot as plt
import pygame

import Constants
from Agent import Agent
from SnakeGame import SnakeGame


def save_individual_plots(
    scores, mean_scores, losses, save_dir, score_ylim=None, mean_score_ylim=None
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure()
    plt.plot(scores, label="Score")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.legend()
    if score_ylim:
        plt.ylim(score_ylim)
    plt.savefig(os.path.join(save_dir, "score_plot.pdf"))
    plt.close()

    plt.figure()
    plt.plot(mean_scores, label="Mean Score (Last 20)")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Score")
    plt.legend()
    if mean_score_ylim:
        plt.ylim(mean_score_ylim)
    plt.savefig(os.path.join(save_dir, "mean_score_plot.pdf"))
    plt.close()

    plt.figure()
    plt.plot(losses, label="Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_plot.pdf"))
    plt.close()


def train(
    agent,
    game,
    episodes,
    show_ui=False,
    checkpoint_interval=50,
    score_ylim=(0, 100),
    mean_score_ylim=(0, 100),
):
    game.show_ui = show_ui
    if not show_ui:
        pygame.display.quit()
    plot_scores = []
    plot_mean_scores = []
    plot_losses = []
    total_score = 0
    record = 0
    scores_window = deque(maxlen=20)  # For running average of the last 20 scores

    for episode in range(episodes):
        steps = 0
        losses = []
        while True:
            old_state = game.get_game_state()
            agent_move = agent.get_action(old_state)
            game_over, reward, total_score = game.play_step(agent_move)
            new_state = game.get_game_state()
            agent_move = agent_move.value

            agent.remember(old_state, agent_move, reward, new_state, game_over)
            loss = agent.train_long_memory()
            losses.append(loss)

            steps += 1

            if game_over:
                break

        game.start_new_game()
        agent.add_games_count()

        plot_scores.append(total_score)
        plot_losses.append(sum(losses) / len(losses) if losses else 0)

        scores_window.append(total_score)
        mean_score = sum(scores_window) / len(scores_window)
        plot_mean_scores.append(mean_score)

        if total_score > record:
            record = total_score

        if episode % checkpoint_interval == 0:
            metrics = {
                "scores": plot_scores,
                "mean_scores": plot_mean_scores,
                "losses": plot_losses,
            }
            agent.save_checkpoint(episode, total_score, record, metrics)

        print(
            f"Game: {agent.n_games} Steps: {steps} Score: {total_score} Record: {record} Loss: {plot_losses[-1]} Mean Score (Last 20): {mean_score}"
        )

    # Save each plot individually as a PDF
    final_model_dir = agent.get_final_model_dir()
    save_individual_plots(
        plot_scores,
        plot_mean_scores,
        plot_losses,
        final_model_dir,
        score_ylim,
        mean_score_ylim,
    )

    # Finalize training by saving the final model and metrics, and deleting checkpoints
    final_metrics = {
        "scores": plot_scores,
        "mean_scores": plot_mean_scores,
        "losses": plot_losses,
        "record": record,
    }
    agent.finalize_training(final_metrics)


def evaluation(final_model_path="model.pth"):
    pygame.display.init()
    agent = Agent(eval_mode=True)
    game = SnakeGame()
    agent.load_model(final_model_path)
    print(f"Loaded final model from {final_model_path}.")
    agent.model.eval()

    scores = []
    record = 0

    for _ in range(100):
        steps = 0
        total_score = 0
        while True:
            old_state = game.get_game_state()
            agent_move = agent.get_action(old_state)
            game_over, reward, total_score = game.play_step(agent_move)

            steps += 1
            if game_over:
                scores.append(total_score)
                if total_score > record:
                    record = total_score
                break

        print(
            f"Game: {agent.n_games} Steps: {steps} Score: {total_score} Record: {record}"
        )

        game.start_new_game()
        agent.add_games_count()

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)

    print(f"Average Score: {avg_score}")
    print(f"Maximum Score: {max_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate the Snake game agent."
    )
    parser.add_argument(
        "--eval-mode", action="store_true", help="Run the agent in evaluation mode."
    )
    args = parser.parse_args()

    if args.eval_mode:
        evaluation("model.pth")
    else:
        pygame.init()
        hyperparameters = {
            "gamma": Constants.GAMMA,
            "max_memory": Constants.MAX_MEMORY,
            "lr": Constants.LR,
            "batch_size": Constants.BATCH_SIZE,
            "training_episodes": Constants.TRAINING_EPISODES,
            "epsilon": Constants.EPSILON,
            "eps_upper": Constants.EPS_UPPER,
        }
        agent = Agent(hyperparameters=hyperparameters)
        game = SnakeGame()

        training_episodes = Constants.TRAINING_EPISODES
        train(
            agent,
            game,
            training_episodes,
            score_ylim=(0, 130),
            mean_score_ylim=(0, 100),
        )
        final_model_dir = agent.get_final_model_dir()
        evaluation(final_model_dir)
