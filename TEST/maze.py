"""Random solvable maze generation using iterative randomized DFS (recursive backtracker).

Grid model: each cell holds 4 walls (N,E,S,W). Carving removes walls between
neighboring cells. The resulting maze is a spanning tree, so any two cells
(including (0,0) -> (rows-1, cols-1)) are connected -> always solvable.

After generation, extra openings can be added to create loops (partial braid maze),
which eliminates long dead ends and makes the maze much easier for DQN to learn.
"""
import random
from typing import List, Tuple

N, E, S, W = 0, 1, 2, 3

_DIRS = [
    (-1, 0, N, S),
    (0, 1, E, W),
    (1, 0, S, N),
    (0, -1, W, E),
]

ACTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class Maze:
    """2D maze. walls[r][c] is a list[bool] of 4 walls indexed by N/E/S/W."""

    def __init__(self, rows: int, cols: int, seed: int | None = None,
                 extra_openings: float = 0.0):
        self.rows = rows
        self.cols = cols
        self.walls: List[List[List[bool]]] = [
            [[True, True, True, True] for _ in range(cols)] for _ in range(rows)
        ]
        self.start: Tuple[int, int] = (0, 0)
        self.goal: Tuple[int, int] = (rows - 1, cols - 1)
        self._rng = random.Random(seed)
        self.carve_history: List[Tuple[int, int]] = []
        self._generate()
        if extra_openings > 0:
            self._add_extra_openings(extra_openings)

    def __repr__(self):
        return f"Maze({self.rows}x{self.cols}, start={self.start}, goal={self.goal})"

    def _generate(self):
        visited = [[False] * self.cols for _ in range(self.rows)]
        stack = [(0, 0)]
        visited[0][0] = True
        self.carve_history.append((0, 0))
        while stack:
            r, c = stack[-1]
            nbrs = []
            for dr, dc, w_cur, w_nbr in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and not visited[nr][nc]:
                    nbrs.append((nr, nc, w_cur, w_nbr))
            if not nbrs:
                stack.pop()
                continue
            nr, nc, w_cur, w_nbr = self._rng.choice(nbrs)
            self.walls[r][c][w_cur] = False
            self.walls[nr][nc][w_nbr] = False
            visited[nr][nc] = True
            self.carve_history.append((nr, nc))
            stack.append((nr, nc))

    def _add_extra_openings(self, ratio: float):
        candidates = []
        for r in range(self.rows):
            for c in range(self.cols):
                if r > 0 and self.walls[r][c][N]:
                    candidates.append((r, c, N, r - 1, c, S))
                if c < self.cols - 1 and self.walls[r][c][E]:
                    candidates.append((r, c, E, r, c + 1, W))
        self._rng.shuffle(candidates)
        n_remove = int(len(candidates) * ratio)
        for r, c, d, nr, nc, opp in candidates[:n_remove]:
            self.walls[r][c][d] = False
            self.walls[nr][nc][opp] = False

    def neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        out = []
        for dr, dc, w_cur, _ in _DIRS:
            if not self.walls[r][c][w_cur]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    out.append((nr, nc))
        return out

    def can_move(self, r: int, c: int, action: int) -> bool:
        return not self.walls[r][c][action]

    def move(self, r: int, c: int, action: int) -> Tuple[int, int]:
        dr, dc = ACTION_DELTAS[action]
        return (r + dr, c + dc)
