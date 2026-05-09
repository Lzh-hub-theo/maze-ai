# Maze: BFS vs DQN — Animated Demo

A fully runnable Python project that:

1. Generates a random solvable 2D maze (start=top-left, goal=bottom-right).
2. Animates **BFS** finding the theoretical shortest path (queue / visited /
   parent backtracking, all rendered frame-by-frame).
3. Trains a **DQN** agent (PyTorch) on a Gym-style maze environment, animating
   every env-step: agent position, trajectory, exploration vs exploitation
   marker, max-Q heatmap, and live reward / loss curves.
4. Plays a side-by-side **comparison** of the BFS path vs the DQN greedy path
   with metrics (path length, search time, training episodes).

## Install

```bash
pip install -r requirements.txt
```

(If you don't need video export, you can skip `imageio-ffmpeg`.)

## Run (one-click)

```bash
python main.py
```

A pygame window opens and plays the four phases in order.

## Controls

| Key            | Action                                  |
| -------------- | --------------------------------------- |
| `SPACE`        | Pause / resume                          |
| `UP` / `DOWN`  | Faster / slower playback                |
| `R`            | Reset / replay current phase            |
| `S`            | Save accumulated frames as video        |
| `ESC` / `X`    | Quit                                    |

During DQN training, `R` skips the rest of the current episode (useful for
long early random episodes).

## Customization

Edit `config.py`:

- `maze_rows`, `maze_cols`, `seed`, `cell_pixels`
- `dqn_episodes`, `dqn_lr`, `dqn_gamma`, `dqn_eps_decay`, `dqn_buffer_size`,
  `dqn_batch_size`, `dqn_target_update`, `dqn_hidden`
- Reward shaping: `reward_goal`, `reward_wall`, `reward_step`
- `render_every` — render only every Nth env-step (faster training)
- `save_video_path` — set to e.g. `"out.mp4"` or `"out.gif"` to enable
  recording; press `S` to flush frames to disk.

## File structure

```
config.py        # all hyperparameters & colors
maze.py          # randomized DFS maze generator
env.py           # Gym-style maze environment
bfs.py           # BFS solver yielding animation frames
dqn_agent.py     # PyTorch DQN + replay buffer + training generator
visualizer.py    # pygame rendering, controls, recording, matplotlib curves
main.py          # one-click orchestrator
requirements.txt
```

## Notes

- Cross-platform (Windows / Linux / macOS) — uses pygame + matplotlib `Agg`.
- BFS is guaranteed optimal on the unweighted grid; DQN may need more episodes
  on larger mazes (try `dqn_episodes=600` for 20×20).
- For video export install `imageio-ffmpeg` and set `save_video_path` before
  running; press `S` at any point to flush captured frames.
