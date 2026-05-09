"""Global configuration: maze size, DQN hyperparameters, rendering options.

All knobs the user is likely to tweak live here. Edit values and re-run main.py.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    maze_rows: int = 15
    maze_cols: int = 15
    cell_pixels: int = 30
    wall_pixels: int = 2
    seed: int = 42
    maze_extra_openings: float = 0.15

    bfs_steps_per_frame: int = 1

    dqn_episodes: int = 500
    dqn_max_steps_factor: int = 3
    dqn_lr: float = 5e-4
    dqn_gamma: float = 0.98
    dqn_eps_start: float = 0.8
    dqn_eps_end: float = 0.01
    dqn_eps_decay_end_frac: float = 0.5
    dqn_batch_size: int = 64
    dqn_buffer_size: int = 30000
    dqn_target_update: int = 10
    dqn_hidden: int = 128
    dqn_train_start: int = 500
    dqn_tau: float = 0.005

    reward_goal: float = 500.0
    reward_wall: float = -3.0
    reward_step: float = -0.5
    reward_approach: float = 2.0
    reward_revisit: float = -0.5

    fps: int = 60
    render_every: int = 50
    train_render_episodes: List[int] = field(default_factory=list)
    save_video_path: str = ""

    window_title: str = "Maze: BFS vs DQN"
    bg_color: Tuple[int, int, int] = (24, 24, 28)
    wall_color: Tuple[int, int, int] = (230, 230, 235)
    start_color: Tuple[int, int, int] = (60, 200, 90)
    goal_color: Tuple[int, int, int] = (230, 70, 70)
    visited_color: Tuple[int, int, int] = (70, 110, 200)
    frontier_color: Tuple[int, int, int] = (250, 200, 60)
    path_color: Tuple[int, int, int] = (255, 130, 60)
    agent_color: Tuple[int, int, int] = (250, 250, 80)
    explore_color: Tuple[int, int, int] = (200, 80, 220)
    exploit_color: Tuple[int, int, int] = (80, 220, 200)
    wall_hit_color: Tuple[int, int, int] = (255, 50, 50)
    revisit_color: Tuple[int, int, int] = (255, 140, 0)


CFG = Config()
