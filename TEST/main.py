"""Main entry point: generate maze → BFS → DQN training → comparison.

Usage:
  python main.py             # full visualization (default)
  python main.py --headless  # headless mode (no pygame window, faster)
"""
import argparse
import sys
import time

from config import CFG
from maze import Maze
from env import MazeEnv
from bfs import solve_steps
from dqn_agent import DQNAgent
from visualizer import Visualizer


def _run_bfs(maze):
    frames = list(solve_steps(maze))
    path = []
    for kind, payload in frames:
        if kind == "done":
            path = payload["path"]
            break
    return path, frames


def main():
    parser = argparse.ArgumentParser(description="BFS+DQN Maze Solver")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    args = parser.parse_args()

    maze = Maze(CFG.maze_rows, CFG.maze_cols, seed=CFG.seed,
                extra_openings=CFG.maze_extra_openings)
    print(f"Maze: {maze.rows}x{maze.cols}  start={maze.start}  goal={maze.goal}")

    bfs_path, bfs_frames = _run_bfs(maze)
    bfs_len = len(bfs_path)
    print(f"BFS shortest path length: {bfs_len}")

    env = MazeEnv(maze)
    agent = DQNAgent(env, bfs_len=bfs_len, seed=CFG.seed)

    if args.headless:
        _run_headless(maze, env, agent, bfs_path, bfs_len)
    else:
        _run_visual(maze, env, agent, bfs_path, bfs_frames, bfs_len)


def _run_headless(maze, env, agent, bfs_path, bfs_len):
    print(f"Training {CFG.dqn_episodes} episodes (headless) ...")
    t0 = time.time()
    best_len = None
    for ep in range(1, CFG.dqn_episodes + 1):
        s = env.reset()
        ep_reward = 0.0
        ep_len = 0
        done = False
        goal_reached = False
        while not done:
            a, _ = agent.select(s)
            s2, r, done, info = env.step(a)
            agent.buffer.push(s, a, r, s2, float(done))
            agent.learn()
            s = s2
            ep_reward += r
            ep_len += 1
            if info.get("goal"):
                goal_reached = True
        agent._decay_eps(ep)
        greedy = agent.greedy_path()
        agent._update_best(greedy)
        best_len = agent.best_greedy_len
        goal_str = "GOAL" if goal_reached else "----"
        best_str = str(best_len) if best_len else "-"
        gap_str = f"  gap={best_len - bfs_len}" if best_len and bfs_len else ""
        print(f"[DQN] ep {ep:>4d}/{CFG.dqn_episodes} "
              f"len={ep_len:>5d} R={ep_reward:+8.1f} "
              f"eps={agent.eps:.3f} {goal_str} best={best_str}{gap_str}")
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s")
    final_greedy = agent.greedy_path()
    fl = len(final_greedy) if final_greedy and final_greedy[-1] == maze.goal else None
    print(f"Final greedy path length: {fl}")
    print(f"BFS optimal path length:  {bfs_len}")
    if fl:
        print(f"Gap from optimal: {fl - bfs_len}")


def _run_visual(maze, env, agent, bfs_path, bfs_frames, bfs_len):
    viz = Visualizer(maze)

    viz.animate_generation()

    bfs_path_result, bfs_expansions, bfs_time = viz.animate_bfs(bfs_frames)
    print(f"BFS done: path={len(bfs_path_result)}  expansions={bfs_expansions}  "
          f"time={bfs_time*1000:.1f}ms")

    t0 = time.time()
    train_gen = agent.train_steps()
    ep_rewards, ep_losses, ep_lengths, dqn_greedy, dqn_best = viz.animate_dqn(
        agent, train_gen)
    dqn_time = time.time() - t0
    print(f"\nDQN training done in {dqn_time:.1f}s")
    print(f"Best greedy path length: {dqn_best}")
    print(f"BFS optimal path length: {bfs_len}")

    if dqn_greedy and dqn_greedy[-1] == maze.goal:
        viz.animate_comparison(bfs_path_result, bfs_time, dqn_greedy,
                               CFG.dqn_episodes, dqn_time)
    else:
        print("DQN did not find goal in greedy path; skipping comparison phase.")

    pygame_quit_safe()


def pygame_quit_safe():
    try:
        import pygame
        pygame.quit()
    except Exception:
        pass


if __name__ == "__main__":
    main()
