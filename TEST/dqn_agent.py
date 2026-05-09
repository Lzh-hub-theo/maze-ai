"""DQN agent (PyTorch) with experience replay + target network + Double DQN.

Key features:
  - 3-layer MLP with LayerNorm for stable training
  - Double DQN to reduce Q-value overestimation
  - Soft target network update (polyak averaging) every learning step
  - Linear epsilon decay with configurable end fraction (faster decay)
  - Detailed training log: GOAL flag, best greedy path, BFS gap
  - 9-dim input: position + walls + goal direction + visit count
"""
import random
from collections import deque
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import MazeEnv
from config import CFG


class QNet(nn.Module):
    def __init__(self, in_dim: int = 9, hidden: int = 128, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.from_numpy(np.array(s, dtype=np.float32)),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.from_numpy(np.array(s2, dtype=np.float32)),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, env: MazeEnv, bfs_len: int | None = None, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        self.env = env
        self.bfs_len = bfs_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = env.obs_dim
        self.qnet = QNet(self.obs_dim, CFG.dqn_hidden, env.n_actions).to(self.device)
        self.tgt = QNet(self.obs_dim, CFG.dqn_hidden, env.n_actions).to(self.device)
        self.tgt.load_state_dict(self.qnet.state_dict())
        for p in self.tgt.parameters():
            p.requires_grad = False
        self.opt = optim.Adam(self.qnet.parameters(), lr=CFG.dqn_lr, eps=1e-4)
        self.buffer = ReplayBuffer(CFG.dqn_buffer_size)
        self.eps = CFG.dqn_eps_start
        self.best_greedy_len: int | None = None

    def select(self, state: np.ndarray):
        if random.random() < self.eps:
            return random.randrange(self.env.n_actions), True
        with torch.no_grad():
            q = self.qnet(torch.from_numpy(state).unsqueeze(0).to(self.device))
        return int(q.argmax(dim=1).item()), False

    def _soft_update(self):
        tau = CFG.dqn_tau
        for p_tgt, p_src in zip(self.tgt.parameters(), self.qnet.parameters()):
            p_tgt.data.mul_(1.0 - tau).add_(p_src.data, alpha=tau)

    def learn(self) -> Optional[float]:
        if len(self.buffer) < CFG.dqn_train_start:
            return None
        s, a, r, s2, d = self.buffer.sample(CFG.dqn_batch_size)
        s, a, r, s2, d = (x.to(self.device) for x in (s, a, r, s2, d))
        q_pred = self.qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_next = self.qnet(s2).argmax(dim=1)
            q_next = self.tgt(s2).gather(1, a_next.unsqueeze(1)).squeeze(1)
            target = r + CFG.dqn_gamma * q_next * (1.0 - d)
        loss = nn.functional.smooth_l1_loss(q_pred, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnet.parameters(), 10.0)
        self.opt.step()
        self._soft_update()
        return float(loss.item())

    def _decay_eps(self, ep: int):
        decay_eps = int(CFG.dqn_episodes * CFG.dqn_eps_decay_end_frac)
        if decay_eps > 0:
            self.eps = max(CFG.dqn_eps_end,
                           CFG.dqn_eps_start - ep * (CFG.dqn_eps_start - CFG.dqn_eps_end) / decay_eps)
        else:
            self.eps = CFG.dqn_eps_end

    def q_heatmap(self) -> np.ndarray:
        rows, cols = self.env.rows, self.env.cols
        states = np.zeros((rows * cols, self.obs_dim), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                states[r * cols + c] = self.env.state_for(r, c)
        with torch.no_grad():
            q = self.qnet(torch.from_numpy(states).to(self.device)).cpu().numpy()
        return q.max(axis=1).reshape(rows, cols)

    def greedy_path(self, max_len: int | None = None) -> List:
        rows, cols = self.env.rows, self.env.cols
        if max_len is None:
            max_len = rows * cols * 4
        path = [self.env.maze.start]
        visited = {self.env.maze.start}
        cur = self.env.maze.start
        for _ in range(max_len):
            if cur == self.env.maze.goal:
                break
            s = self.env.state_for(*cur)
            with torch.no_grad():
                q = self.qnet(torch.from_numpy(s).unsqueeze(0).to(self.device))[0].cpu().numpy()
            order = np.argsort(-q)
            moved = False
            for a in order:
                a = int(a)
                if self.env.maze.can_move(cur[0], cur[1], a):
                    nxt = self.env.maze.move(cur[0], cur[1], a)
                    if nxt not in visited:
                        cur = nxt
                        path.append(cur)
                        visited.add(nxt)
                        moved = True
                        break
            if not moved:
                best_nxt = None
                best_q = -float("inf")
                for a in range(self.env.n_actions):
                    if self.env.maze.can_move(cur[0], cur[1], a):
                        nxt = self.env.maze.move(cur[0], cur[1], a)
                        if q[a] > best_q:
                            best_q = q[a]
                            best_nxt = nxt
                if best_nxt is not None:
                    cur = best_nxt
                    path.append(cur)
                else:
                    break
        return path

    def _update_best(self, greedy_path):
        if greedy_path and greedy_path[-1] == self.env.maze.goal:
            gl = len(greedy_path)
            if self.best_greedy_len is None or gl < self.best_greedy_len:
                self.best_greedy_len = gl

    def _log_line(self, ep, ep_len, ep_reward, goal_reached) -> str:
        parts = [
            f"[DQN] ep {ep:>4d}/{CFG.dqn_episodes}",
            f"len={ep_len:>5d}",
            f"R={ep_reward:+8.1f}",
            f"eps={self.eps:.3f}",
        ]
        parts.append("GOAL" if goal_reached else "----")
        if self.best_greedy_len is not None:
            parts.append(f"best={self.best_greedy_len}")
            if self.bfs_len is not None:
                gap = self.best_greedy_len - self.bfs_len
                parts.append(f"gap={gap}")
        else:
            parts.append("best=-")
        return "  ".join(parts)

    def train_steps(self):
        for ep in range(1, CFG.dqn_episodes + 1):
            s = self.env.reset()
            ep_reward = 0.0
            ep_len = 0
            traj = [self.env.pos]
            done = False
            goal_reached = False
            losses = []
            while not done:
                a, was_explore = self.select(s)
                s2, r, done, info = self.env.step(a)
                self.buffer.push(s, a, r, s2, float(done))
                loss = self.learn()
                if loss is not None:
                    losses.append(loss)
                s = s2
                ep_reward += r
                ep_len += 1
                traj.append(self.env.pos)
                if info.get("goal"):
                    goal_reached = True
                yield {
                    "phase": "step",
                    "episode": ep,
                    "step": ep_len,
                    "pos": self.env.pos,
                    "action": a,
                    "was_explore": was_explore,
                    "reward": r,
                    "done": done,
                    "loss": loss,
                    "ep_reward": ep_reward,
                    "eps": self.eps,
                    "trajectory": list(traj),
                    "goal_reached": goal_reached,
                    "hit_wall": info.get("hit_wall", False),
                    "revisit": info.get("revisit", False),
                }
            self._decay_eps(ep)
            greedy = self.greedy_path()
            self._update_best(greedy)
            yield {
                "phase": "episode_end",
                "episode": ep,
                "ep_reward": ep_reward,
                "ep_len": ep_len,
                "eps": self.eps,
                "avg_loss": float(np.mean(losses)) if losses else None,
                "trajectory": traj,
                "heatmap": self.q_heatmap(),
                "greedy_path": greedy,
                "goal_reached": goal_reached,
                "best_greedy_len": self.best_greedy_len,
            }
