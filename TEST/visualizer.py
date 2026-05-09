"""Pygame-based visualizer for maze generation, BFS, DQN training, and comparison.

Controls (active in any animation phase):
  SPACE      play/pause
  UP / DOWN  speed up / slow down (multiplies frame rate)
  R          reset current phase (replay from beginning)
  S          save the current accumulated frames to MP4/GIF (uses CFG.save_video_path)
  ESC / X    quit
"""
import os
import sys
import time
from typing import List, Optional

import numpy as np
import pygame

from config import CFG
from maze import Maze

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_surface(fig) -> pygame.Surface:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf).reshape(h, w, 4)
    return pygame.image.frombuffer(arr.tobytes(), (w, h), "RGBA")


def _heatmap_surface(values: np.ndarray, cell_px: int) -> pygame.Surface:
    rows, cols = values.shape
    vmin, vmax = float(values.min()), float(values.max())
    rng = max(1e-6, vmax - vmin)
    t = (values - vmin) / rng
    r_ch = np.clip(255 * t, 0, 255).astype(np.uint8)
    g_ch = np.clip(255 * t * t, 0, 255).astype(np.uint8)
    b_ch = np.clip(255 * (1 - t), 0, 255).astype(np.uint8)
    img = np.stack([r_ch, g_ch, b_ch], axis=-1)
    img = np.repeat(np.repeat(img, cell_px, axis=0), cell_px, axis=1)
    surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
    return surf


class Visualizer:
    def __init__(self, maze: Maze):
        pygame.init()
        self.maze = maze
        self.cell = CFG.cell_pixels
        self.maze_w = maze.cols * self.cell
        self.maze_h = maze.rows * self.cell
        self.right_w = 520
        self.win_w = self.maze_w + self.right_w + 30
        self.win_h = max(self.maze_h, 720) + 80
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption(CFG.window_title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 18)
        self.bigfont = pygame.font.Font(None, 26)
        self.paused = False
        self.speed = 1.0
        self.reset_flag = False
        self._quit_flag = False
        self.record_frames: List[np.ndarray] = []
        self.recording = bool(CFG.save_video_path)

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self._quit_flag = True
                self._quit()
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_x):
                    self._quit_flag = True
                    self._quit()
                elif ev.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif ev.key == pygame.K_UP:
                    self.speed = min(16.0, self.speed * 1.5)
                elif ev.key == pygame.K_DOWN:
                    self.speed = max(0.1, self.speed / 1.5)
                elif ev.key == pygame.K_r:
                    self.reset_flag = True
                elif ev.key == pygame.K_s:
                    self._save_recording()

    def _quit(self):
        pygame.quit()
        sys.exit(0)

    def _tick(self, base_fps: int = None):
        base = base_fps if base_fps is not None else CFG.fps
        self.clock.tick(int(max(1, base * self.speed)))
        if self.recording:
            arr = pygame.surfarray.array3d(self.screen)
            arr = np.transpose(arr, (1, 0, 2))
            self.record_frames.append(arr)
        while self.paused:
            self._handle_events()
            if self.reset_flag:
                self.paused = False
                return
            pygame.time.wait(50)
        self._handle_events()

    def _draw_status(self, lines: List[str], x: int, y: int, color=(220, 220, 220)):
        for i, ln in enumerate(lines):
            self.screen.blit(self.font.render(ln, True, color), (x, y + i * 20))

    def _cell_rect(self, r: int, c: int):
        return pygame.Rect(c * self.cell + 10, r * self.cell + 10, self.cell, self.cell)

    def _draw_maze(self, walls_carved_set=None,
                   visited=None, frontier=None, current=None,
                   path=None, agent=None, trajectory=None,
                   show_walls: bool = True,
                   revisit_cells=None, wall_hit_pos=None):
        pygame.draw.rect(self.screen, (15, 15, 18),
                         (5, 5, self.maze_w + 10, self.maze_h + 10))
        if visited:
            for (r, c) in visited:
                pygame.draw.rect(self.screen, CFG.visited_color, self._cell_rect(r, c))
        if frontier:
            for (r, c) in frontier:
                pygame.draw.rect(self.screen, CFG.frontier_color, self._cell_rect(r, c))
        if revisit_cells:
            for (r, c) in revisit_cells:
                rect = self._cell_rect(r, c).inflate(-self.cell // 3, -self.cell // 3)
                pygame.draw.rect(self.screen, CFG.revisit_color, rect, border_radius=2)
        if trajectory:
            for i, (r, c) in enumerate(trajectory):
                fade = max(60, 220 - 4 * (len(trajectory) - i))
                col = (fade, fade // 2, 30)
                rect = self._cell_rect(r, c).inflate(-self.cell // 2, -self.cell // 2)
                pygame.draw.rect(self.screen, col, rect)
        if path:
            for (r, c) in path:
                rect = self._cell_rect(r, c).inflate(-self.cell // 3, -self.cell // 3)
                pygame.draw.rect(self.screen, CFG.path_color, rect, border_radius=3)
        sr, sc = self.maze.start
        gr, gc = self.maze.goal
        pygame.draw.rect(self.screen, CFG.start_color, self._cell_rect(sr, sc))
        pygame.draw.rect(self.screen, CFG.goal_color, self._cell_rect(gr, gc))
        if current is not None:
            r, c = current
            pygame.draw.rect(self.screen, (255, 255, 255), self._cell_rect(r, c), 2)
        if wall_hit_pos is not None:
            r, c = wall_hit_pos
            rect = self._cell_rect(r, c).inflate(-2, -2)
            pygame.draw.rect(self.screen, CFG.wall_hit_color, rect, 3)
        if agent is not None:
            r, c = agent
            cx = c * self.cell + 10 + self.cell // 2
            cy = r * self.cell + 10 + self.cell // 2
            pygame.draw.circle(self.screen, CFG.agent_color, (cx, cy), max(4, self.cell // 3))
        if show_walls:
            self._draw_walls(walls_carved_set)

    def _draw_walls(self, carved_only=None):
        wt = CFG.wall_pixels
        col = CFG.wall_color
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                if carved_only is not None and (r, c) not in carved_only:
                    continue
                x = c * self.cell + 10
                y = r * self.cell + 10
                w = self.cell
                if self.maze.walls[r][c][0]:
                    pygame.draw.line(self.screen, col, (x, y), (x + w, y), wt)
                if self.maze.walls[r][c][1]:
                    pygame.draw.line(self.screen, col, (x + w, y), (x + w, y + w), wt)
                if self.maze.walls[r][c][2]:
                    pygame.draw.line(self.screen, col, (x, y + w), (x + w, y + w), wt)
                if self.maze.walls[r][c][3]:
                    pygame.draw.line(self.screen, col, (x, y), (x, y + w), wt)

    def _draw_small_maze(self, cell_px: int, path_cells=None) -> pygame.Surface:
        small_w = self.maze.cols * cell_px
        small_h = self.maze.rows * cell_px
        surf = pygame.Surface((small_w + 4, small_h + 4))
        surf.fill((15, 15, 18))
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                x, y = c * cell_px + 2, r * cell_px + 2
                w = cell_px
                if self.maze.walls[r][c][0]:
                    pygame.draw.line(surf, CFG.wall_color, (x, y), (x + w, y), 1)
                if self.maze.walls[r][c][1]:
                    pygame.draw.line(surf, CFG.wall_color, (x + w, y), (x + w, y + w), 1)
                if self.maze.walls[r][c][2]:
                    pygame.draw.line(surf, CFG.wall_color, (x, y + w), (x + w, y + w), 1)
                if self.maze.walls[r][c][3]:
                    pygame.draw.line(surf, CFG.wall_color, (x, y), (x, y + w), 1)
        sr, sc = self.maze.start
        gr, gc = self.maze.goal
        pygame.draw.rect(surf, CFG.start_color,
                         (sc * cell_px + 2, sr * cell_px + 2, cell_px, cell_px))
        pygame.draw.rect(surf, CFG.goal_color,
                         (gc * cell_px + 2, gr * cell_px + 2, cell_px, cell_px))
        if path_cells:
            for (r, c) in path_cells:
                rect = pygame.Rect(c * cell_px + 2 + cell_px // 4,
                                   r * cell_px + 2 + cell_px // 4,
                                   cell_px // 2, cell_px // 2)
                pygame.draw.rect(surf, CFG.path_color, rect)
        return surf

    def animate_generation(self):
        history = self.maze.carve_history
        i = 0
        while i <= len(history):
            self.reset_flag = False
            self.screen.fill(CFG.bg_color)
            visited = set(history[:i])
            self._draw_maze(walls_carved_set=visited, visited=visited)
            self._draw_status([
                "Phase: Maze Generation",
                f"Cells carved: {i}/{len(history)}",
                "SPACE pause | UP/DOWN speed | R reset | S save | ESC quit",
                f"Speed x{self.speed:.2f}  Paused={self.paused}",
            ], self.maze_w + 25, 10)
            pygame.display.flip()
            self._tick(60)
            if self.reset_flag:
                i = 0
                continue
            i += 1
        for _ in range(20):
            self._tick(60)

    def animate_bfs(self, frames):
        frames = list(frames)
        t0 = time.time()
        idx = 0
        final_path = []
        expansions = 0
        visited_count = 0
        while idx < len(frames):
            self.reset_flag = False
            kind, payload = frames[idx]
            self.screen.fill(CFG.bg_color)
            if kind == "expand":
                self._draw_maze(
                    visited=payload["visited"],
                    frontier=payload["frontier"],
                    current=payload["current"],
                )
                visited_count = len(payload["visited"])
            elif kind == "path_step":
                self._draw_maze(
                    visited=payload["visited"],
                    path=payload["path_so_far"],
                )
            elif kind == "done":
                final_path = payload["path"]
                expansions = payload["expansions"]
                self._draw_maze(visited=payload["visited"], path=final_path)
            elapsed = time.time() - t0
            self._draw_status([
                "Phase: BFS Search",
                f"Frame: {idx + 1}/{len(frames)}  ({kind})",
                f"Visited: {visited_count}  Expansions: {expansions}",
                f"Path length so far: {len(final_path)}",
                f"Elapsed: {elapsed:.2f}s",
                "SPACE pause | UP/DOWN speed | R reset | S save",
            ], self.maze_w + 25, 10)
            pygame.display.flip()
            self._tick(CFG.fps)
            if self.reset_flag:
                idx = 0
                t0 = time.time()
                final_path, expansions, visited_count = [], 0, 0
                continue
            idx += 1
        for _ in range(30):
            self._tick(60)
        return final_path, expansions, time.time() - t0

    def animate_dqn(self, agent, train_gen):
        ep_rewards: List[float] = []
        ep_losses: List[float] = []
        ep_lengths: List[int] = []
        last_heatmap: Optional[np.ndarray] = None
        last_greedy: List = []
        best_len = None
        bfs_len = agent.bfs_len
        wall_hit_flash = None
        revisit_flash = None

        fig, axes = plt.subplots(2, 1, figsize=(5.0, 3.6), dpi=100)
        fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.12, hspace=0.55)

        steps_seen = 0
        for update in train_gen:
            if update["phase"] == "step":
                steps_seen += 1
                if steps_seen % CFG.render_every != 0 and not update["done"]:
                    continue
                self.screen.fill(CFG.bg_color)
                if update.get("hit_wall"):
                    wall_hit_flash = update["pos"]
                else:
                    wall_hit_flash = None
                if update.get("revisit"):
                    revisit_flash = [update["pos"]]
                else:
                    revisit_flash = None
                self._draw_maze(
                    trajectory=update["trajectory"],
                    agent=update["pos"],
                    path=last_greedy if last_greedy else None,
                    wall_hit_pos=wall_hit_flash,
                    revisit_cells=revisit_flash,
                )
                if last_heatmap is not None:
                    hm = _heatmap_surface(last_heatmap, max(8, self.cell // 2))
                    self.screen.blit(hm, (self.maze_w + 25, 260))
                    self.screen.blit(self.font.render("max-Q heatmap (blue=low, yellow=high)", True, (200, 200, 200)),
                                     (self.maze_w + 25, 260 + hm.get_height() + 4))
                self._render_curves(fig, axes, ep_rewards, ep_losses)
                marker_color = CFG.explore_color if update["was_explore"] else CFG.exploit_color
                if update.get("hit_wall"):
                    marker_color = CFG.wall_hit_color
                elif update.get("revisit"):
                    marker_color = CFG.revisit_color
                pygame.draw.circle(self.screen, marker_color, (self.maze_w + 35, 18), 7)
                action_str = ['N', 'E', 'S', 'W'][update['action']]
                mode = 'EXPLORE' if update['was_explore'] else 'exploit'
                if update.get("hit_wall"):
                    mode = 'WALL_HIT'
                elif update.get("revisit"):
                    mode = 'REVISIT'
                goal_flag = "GOAL" if update.get("goal_reached") else "----"
                best_str = str(best_len) if best_len else "-"
                gap_str = ""
                if best_len and bfs_len:
                    gap_str = f"  gap={best_len - bfs_len}"
                self._draw_status([
                    "Phase: DQN Training",
                    f"Episode: {update['episode']}/{CFG.dqn_episodes}",
                    f"Step: {update['step']}  Action: {action_str}  ({mode})",
                    f"Eps: {update['eps']:.3f}   Reward(ep): {update['ep_reward']:.1f}",
                    f"Last loss: {update['loss'] if update['loss'] is not None else '-'}",
                    f"Goal: {goal_flag}  Best: {best_str}{gap_str}",
                    "SPACE pause | UP/DOWN speed | R skip | S save",
                ], self.maze_w + 50, 10)
                pygame.display.flip()
                self._tick(CFG.fps)
                if self.reset_flag:
                    self.reset_flag = False
                    for nxt in train_gen:
                        if nxt["phase"] == "episode_end":
                            update = nxt
                            break
            if update["phase"] == "episode_end":
                ep_rewards.append(update["ep_reward"])
                ep_lengths.append(update["ep_len"])
                if update["avg_loss"] is not None:
                    ep_losses.append(update["avg_loss"])
                last_heatmap = update["heatmap"]
                last_greedy = update["greedy_path"]
                if last_greedy and last_greedy[-1] == self.maze.goal:
                    if best_len is None or len(last_greedy) < best_len:
                        best_len = len(last_greedy)
                goal_str = "GOAL" if update["goal_reached"] else "----"
                best_str = str(best_len) if best_len else "-"
                gap_str = ""
                if best_len and bfs_len:
                    gap_str = f"  gap={best_len - bfs_len}"
                print(f"[DQN] ep {update['episode']:>4d}/{CFG.dqn_episodes} "
                      f"len={update['ep_len']:>5d} R={update['ep_reward']:+8.1f} "
                      f"eps={update['eps']:.3f} {goal_str} best={best_str}{gap_str}")
        plt.close(fig)
        return ep_rewards, ep_losses, ep_lengths, last_greedy, best_len

    def _render_curves(self, fig, axes, rewards, losses):
        ax1, ax2 = axes
        ax1.clear(); ax2.clear()
        ax1.set_title("Episode Reward", fontsize=10)
        ax1.plot(rewards, color="#ffaa55")
        if rewards:
            ax1.axhline(y=0, color='#666', linestyle='--', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        ax2.set_title("Avg Loss / Episode", fontsize=10)
        ax2.plot(losses, color="#55aaff")
        ax2.grid(True, alpha=0.3)
        fig.patch.set_facecolor("#181820")
        for ax in axes:
            ax.set_facecolor("#222230")
            ax.tick_params(colors="#bbb", labelsize=8)
            for s in ax.spines.values():
                s.set_color("#666")
            ax.title.set_color("#ddd")
        surf = _fig_to_surface(fig)
        self.screen.blit(surf, (self.maze_w + 25, self.win_h - surf.get_height() - 10))

    def animate_comparison(self, bfs_path, bfs_time, dqn_path, dqn_episodes,
                           dqn_train_time):
        n = max(len(bfs_path), len(dqn_path))
        small_cell = max(8, self.cell // 2)
        i = 0
        while i < n + 30:
            self.reset_flag = False
            self.screen.fill(CFG.bg_color)
            bp = bfs_path[: min(i + 1, len(bfs_path))]
            self._draw_maze(path=bp, agent=bp[-1] if bp else None)
            dp = dqn_path[: min(i + 1, len(dqn_path))]
            dqn_surf = self._draw_small_maze(small_cell, path_cells=dp)
            self.screen.blit(dqn_surf, (self.maze_w + 25, 260))
            self.screen.blit(self.bigfont.render("BFS (left)  vs  DQN (right)", True, (250, 250, 250)),
                             (self.maze_w + 25, 10))
            self._draw_status([
                f"BFS path length:  {len(bfs_path)}",
                f"BFS search time:  {bfs_time*1000:.1f} ms",
                f"DQN path length:  {len(dqn_path) if dqn_path else 'N/A'}",
                f"DQN episodes:     {dqn_episodes}",
                f"DQN train time:   {dqn_train_time:.1f} s",
                f"Optimal? {'YES' if dqn_path and len(dqn_path) == len(bfs_path) else 'no'}",
                "",
                "SPACE pause | UP/DOWN speed | R reset | S save | ESC quit",
            ], self.maze_w + 25, 60)
            pygame.display.flip()
            self._tick(15)
            if self.reset_flag:
                i = 0
                continue
            i += 1
        hold_frames = 0
        while hold_frames < 300:
            self._handle_events()
            if self._quit_flag:
                return
            self.clock.tick(30)
            hold_frames += 1

    def _save_recording(self):
        path = CFG.save_video_path
        if not path or not self.record_frames:
            print("[save] nothing to save (set CFG.save_video_path and run with recording).")
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            import imageio.v2 as imageio
        except Exception:
            print("[save] imageio not installed; pip install imageio imageio-ffmpeg")
            return
        print(f"[save] writing {len(self.record_frames)} frames to {path} ...")
        if ext in (".gif",):
            imageio.mimsave(path, self.record_frames, fps=20)
        else:
            imageio.mimsave(path, self.record_frames, fps=30, codec="libx264", quality=7)
        print(f"[save] done -> {path}")
