"""Gym-style environment wrapping a Maze.

Observation: 9-dim float32 vector:
  [norm_row, norm_col, wall_N, wall_E, wall_S, wall_W, goal_dr, goal_dc, visit_norm]
  - position normalized to [0,1]
  - wall flags: 1.0 = wall present, 0.0 = open
  - goal direction: normalized (goal_row - cur_row, goal_col - cur_col)
  - visit_norm: min(visit_count / 5.0, 1.0) — how many times current cell was visited

Actions: 0=N, 1=E, 2=S, 3=W.

Reward structure:
  - reward_goal:      large positive on reaching goal (dominates all other rewards)
  - reward_wall:      penalty for hitting a wall (stays in place)
  - reward_step:      small time penalty per step
  - reward_approach:  bonus proportional to Manhattan distance decrease toward goal
  - reward_revisit:   penalty for stepping onto a previously visited cell
"""
import numpy as np
from maze import Maze
from config import CFG


class MazeEnv:
    n_actions = 4
    obs_dim = 9

    def __init__(self, maze: Maze):
        self.maze = maze
        self.rows = maze.rows
        self.cols = maze.cols
        self.pos = maze.start
        self.steps = 0
        self.max_steps = CFG.dqn_max_steps_factor * self.rows * self.cols
        self._prev_dist = self._manhattan(self.pos)
        self._visit_counts: dict = {}

    def _manhattan(self, pos):
        return abs(pos[0] - self.maze.goal[0]) + abs(pos[1] - self.maze.goal[1])

    def reset(self):
        self.pos = self.maze.start
        self.steps = 0
        self._prev_dist = self._manhattan(self.pos)
        self._visit_counts = {self.pos: 1}
        return self._obs()

    def step(self, action: int):
        self.steps += 1
        r, c = self.pos
        if self.maze.can_move(r, c, action):
            self.pos = self.maze.move(r, c, action)
            visit_count = self._visit_counts.get(self.pos, 0)
            self._visit_counts[self.pos] = visit_count + 1
            if self.pos == self.maze.goal:
                return self._obs(), CFG.reward_goal, True, {
                    "hit_wall": False, "goal": True, "revisit": False}
            new_dist = self._manhattan(self.pos)
            approach = CFG.reward_approach * (self._prev_dist - new_dist)
            self._prev_dist = new_dist
            revisit = visit_count > 0
            reward = CFG.reward_step + approach
            if revisit:
                reward += CFG.reward_revisit
            done = self.steps >= self.max_steps
            return self._obs(), reward, done, {
                "hit_wall": False, "goal": False, "revisit": revisit}
        done = self.steps >= self.max_steps
        return self._obs(), CFG.reward_wall, done, {
            "hit_wall": True, "goal": False, "revisit": False}

    def _obs(self):
        r, c = self.pos
        gr, gc = self.maze.goal
        norm_r = r / max(1, self.rows - 1)
        norm_c = c / max(1, self.cols - 1)
        walls = [float(self.maze.walls[r][c][d]) for d in range(4)]
        goal_dr = (gr - r) / max(1, self.rows - 1)
        goal_dc = (gc - c) / max(1, self.cols - 1)
        visit_norm = min(self._visit_counts.get(self.pos, 1) / 5.0, 1.0)
        return np.array(
            [norm_r, norm_c] + walls + [goal_dr, goal_dc, visit_norm],
            dtype=np.float32)

    def state_for(self, r: int, c: int):
        gr, gc = self.maze.goal
        norm_r = r / max(1, self.rows - 1)
        norm_c = c / max(1, self.cols - 1)
        walls = [float(self.maze.walls[r][c][d]) for d in range(4)]
        goal_dr = (gr - r) / max(1, self.rows - 1)
        goal_dc = (gc - c) / max(1, self.cols - 1)
        return np.array(
            [norm_r, norm_c] + walls + [goal_dr, goal_dc, 0.0],
            dtype=np.float32)
