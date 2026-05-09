"""BFS solver that yields step-by-step state for animation.

solve_steps() returns a list of frames; each frame is a dict describing the
frontier, visited set, and current expansion. After exhaustion, the parent
dict is used to reconstruct the path (also yielded as backtracking frames).
"""
from collections import deque
from typing import Dict, List, Tuple
from maze import Maze

Cell = Tuple[int, int]


def solve_steps(maze: Maze):
    """Yield BFS animation frames.

    Each yielded item is a tuple (kind, payload):
      - ("expand", {current, visited:set, frontier:list}) per pop
      - ("path_step", {path_so_far:list}) for each step of backtracking
      - ("done", {path:list, visited:set, expansions:int})
    """
    start, goal = maze.start, maze.goal
    parent: Dict[Cell, Cell] = {start: None}
    visited = {start}
    queue: deque = deque([start])
    expansions = 0
    found = False

    while queue:
        cur = queue.popleft()
        expansions += 1
        # Snapshot frontier as a list (for rendering)
        yield ("expand", {
            "current": cur,
            "visited": set(visited),
            "frontier": list(queue),
            "parent": dict(parent),
        })
        if cur == goal:
            found = True
            break
        for nb in maze.neighbors(*cur):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                queue.append(nb)

    # Reconstruct path via parent backtracking
    path: List[Cell] = []
    if found:
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

    # Animate the backtracking step by step (drawn from start->goal for clarity)
    for i in range(1, len(path) + 1):
        yield ("path_step", {"path_so_far": path[:i], "visited": visited})

    yield ("done", {"path": path, "visited": visited, "expansions": expansions})
