import random
from collections import deque

WALL = "#"
SPACE = " "
PELLET = "."
CAPSULE = "o"
PACMAN = "P"
GHOST = "G"
def _neighbors4(grid, r, c):
    H, W = len(grid), len(grid[0])
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        rr, cc = r+dr, c+dc
        if 0 <= rr < H and 0 <= cc < W:
            yield rr, cc

def _degree(grid, r, c):
    return sum(grid[rr][cc] == " " for rr, cc in _neighbors4(grid, r, c))

def enrich_loops(grid, prob=0.18):
    """
    Probabilistically turns some degree-2 corridor tiles into degree-3 by
    knocking a neighboring wall that leads to another corridor. This
    increases alternative routes without making the whole map a big room.
    """
    import random
    H, W = len(grid), len(grid[0])
    for r in range(2, H-2):
        for c in range(2, W-2):
            if grid[r][c] != " ":
                continue
            if _degree(grid, r, c) == 2 and random.random() < prob:
                # try to open one extra side into a corridor two steps away
                dirs = [(-1,0),(1,0),(0,-1),(0,1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    wr, wc = r+dr, c+dc
                    rr, cc = r+2*dr, c+2*dc
                    if grid[wr][wc] == "#" and grid[rr][cc] == " ":
                        grid[wr][wc] = " "
                        break

def carve_avenues(grid, row_stride=6, col_stride=8, row_offset=0, col_offset=0):
    """
    Carves long, straight corridors at regular intervals to create
    reliable escape lanes. Keeps single-tile corridors.
    """
    H, W = len(grid), len(grid[0])

    # horizontal avenues
    r = max(1, row_offset)
    while r < H-1:
        for c in range(1, W-1):
            grid[r][c] = " "
        r += row_stride

    # vertical avenues
    c = max(1, col_offset)
    while c < W-1:
        for r in range(1, H-1):
            grid[r][c] = " "
        c += col_stride

def add_plazas(grid, rate=0.015):
    """
    Very small 2x2 open pockets to give dodge space at some intersections.
    Keep it low so the map doesn't get 'roomy'.
    """
    import random
    H, W = len(grid), len(grid[0])
    for r in range(2, H-3):
        for c in range(2, W-3):
            if random.random() < rate:
                # only create if at a corridor cross to avoid dead ends
                if grid[r][c] == " " and _degree(grid, r, c) >= 2:
                    for rr in (r, r+1):
                        for cc in (c, c+1):
                            grid[rr][cc] = " "

def place_capsules_square(grid, num_per_side=2, search_radius=4):
    """
    Places capsules ('o') in a num_per_side x num_per_side square grid,
    equally spaced over the playable interior. Each target is nudged to
    the nearest corridor tile if it lands on a wall.

    num_per_side=2 -> 4 capsules (classic corner-like layout).
    """
    H, W = len(grid), len(grid[0])
    inb = lambda r,c: 0 <= r < H and 0 <= c < W

    # interior bounds (ignore outer walls)
    r_min, r_max = 1, H - 2
    c_min, c_max = 1, W - 2

    # equal spacing grid lines
    def linspace_int(lo, hi, n):
        # n positions between lo..hi (inclusive) spaced as evenly as possible
        # use integer rounding so they fall onto cells
        span = hi - lo
        return [lo + round(i * span / (n - 1)) for i in range(n)] if n > 1 else [(lo + hi)//2]

    rows = linspace_int(r_min + 1, r_max - 1, num_per_side)
    cols = linspace_int(c_min + 1, c_max - 1, num_per_side)

    # BFS search to nudge to nearest corridor
    from collections import deque
    def nearest_space(sr, sc):
        if inb(sr, sc) and grid[sr][sc] == " ":
            return (sr, sc)
        q = deque([(sr, sc, 0)])
        seen = {(sr, sc)}
        while q:
            r, c, d = q.popleft()
            if d > search_radius:  # cap search
                break
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r+dr, c+dc
                if inb(rr, cc) and (rr, cc) not in seen:
                    if grid[rr][cc] == " ":
                        return (rr, cc)
                    seen.add((rr, cc))
                    q.append((rr, cc, d+1))
        return None

    # place capsules
    placed = 0
    for r in rows:
        for c in cols:
            spot = nearest_space(r, c)
            if spot:
                rr, cc = spot
                grid[rr][cc] = "o"
                placed += 1
    return placed

def generate_pacman_map(width=31, height=21, num_capsules=4, num_ghosts=4, symmetry="vertical", seed=None):
    """
    Generates an ASCII map with:
      - Single-width corridors
      - No dead ends (all corridor tiles have degree >= 2)
      - Pac-Man at center top, ghosts at bottom
      - Capsules evenly spaced
    width/height should be odd to align corridors cleanly (enforced if not).
    symmetry: None | 'vertical' | 'horizontal'
    """
    if seed is not None:
        random.seed(seed)

    # Force odd dimensions for clean walls between cells
    if width % 2 == 0: width += 1
    if height % 2 == 0: height += 1

    # 1) Full walls
    grid = [[WALL for _ in range(width)] for _ in range(height)]

    # Helpers
    def inb(r, c): return 0 <= r < height and 0 <= c < width
    def neighbors4(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            rr, cc = r+dr, c+dc
            if inb(rr, cc):
                yield rr, cc

    # 2) Carve a base perfect maze on odd cells (DFS backtracker)
    def neighbors2(r, c):
        for dr, dc in [(-2,0),(2,0),(0,-2),(0,2)]:
            rr, cc = r+dr, c+dc
            wr, wc = r + dr//2, c + dc//2
            if 1 <= rr < height-1 and 1 <= cc < width-1:
                yield rr, cc, wr, wc

    sr = random.randrange(1, height, 2)
    sc = random.randrange(1, width, 2)
    grid[sr][sc] = SPACE
    stack = [(sr, sc)]
    seen = {(sr, sc)}
    while stack:
        r, c = stack[-1]
        cand = [(rr, cc, wr, wc) for rr, cc, wr, wc in neighbors2(r, c) if (rr, cc) not in seen]
        if not cand:
            stack.pop()
            continue
        rr, cc, wr, wc = random.choice(cand)
        grid[wr][wc] = SPACE
        grid[rr][cc] = SPACE
        seen.add((rr, cc))
        stack.append((rr, cc))

    # 3) Remove dead ends by adding loops until every corridor has deg >= 2
    def degree(r, c):
        return sum(1 for rr, cc in neighbors4(r, c) if grid[rr][cc] == SPACE)
    # Iteratively open walls next to dead-ends
    changed = True
    while changed:
        changed = False
        for r in range(1, height-1):
            for c in range(1, width-1):
                if grid[r][c] == SPACE and degree(r, c) == 1:
                    # Try to connect through a wall to create an extra exit
                    # Prefer knocking a wall that leads to another corridor two steps away
                    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
                    random.shuffle(dirs)
                    opened = False
                    for dr, dc in dirs:
                        wr, wc = r+dr, c+dc
                        rr, cc = r+2*dr, c+2*dc
                        if inb(rr, cc) and grid[wr][wc] == WALL and grid[rr][cc] == SPACE:
                            grid[wr][wc] = SPACE
                            opened = True
                            changed = True
                            break
                    # If none available (rare), just open any adjacent wall inside bounds
                    if not opened:
                        for dr, dc in dirs:
                            wr, wc = r+dr, c+dc
                            if 1 <= wr < height-1 and 1 <= wc < width-1 and grid[wr][wc] == WALL:
                                grid[wr][wc] = SPACE
                                changed = True
                                break
    carve_avenues(grid, row_stride=6, col_stride=8, row_offset=3, col_offset=width//2 % 3)
    enrich_loops(grid, prob=0.20)
    add_plazas(grid, rate=0.00)
    
    # 4) Optional symmetry (after loops so corridors remain single-wide)
    if symmetry in ("vertical", "horizontal"):
        if symmetry == "vertical":
            mid = width // 2
            for r in range(height):
                for c in range(mid):
                    grid[r][width-1-c] = grid[r][c]
        elif symmetry == "horizontal":
            mid = height // 2
            for r in range(mid):
                grid[height-1-r] = grid[r][:]

    # 5) Place Pac-Man (center-top corridor)
    pac_c = width // 2
    pac_r = None
    for r in range(1, height//3 + 1):
        if grid[r][pac_c] == SPACE:
            pac_r = r
            break
    # If the center column is blocked near the top, scan nearby columns
    if pac_r is None:
        for r in range(1, height//3 + 1):
            for off in range(1, pac_c):
                for c in (pac_c - off, pac_c + off):
                    if 1 <= c < width-1 and grid[r][c] == SPACE:
                        pac_r, pac_c = r, c
                        break
                if pac_r is not None:
                    break
            if pac_r is not None:
                break
    grid[pac_r][pac_c] = PACMAN

    # 6) Place Ghosts (bottom cluster around center on a corridor row)
    base_row = height - 3
    # Find a corridor row near the bottom
    ghost_row = None
    for r in range(base_row, height//2, -1):
        if any(grid[r][c] == SPACE for c in range(1, width-1)):
            ghost_row = r
            break
    if ghost_row is None:
        ghost_row = height - 3
    # Place centered ghosts, skipping walls
    centers = [pac_c]
    i = 1
    while len(centers) < num_ghosts:
        if pac_c - i > 1: centers.append(pac_c - i)
        if len(centers) < num_ghosts and pac_c + i < width-1: centers.append(pac_c + i)
        i += 2  # spread with gaps so they don't overlap
    placed = 0
    for c in centers:
        if placed >= num_ghosts: break
        if grid[ghost_row][c] == SPACE:
            grid[ghost_row][c] = GHOST
            placed += 1
    # Fallback: scan row for spaces if some positions were walls
    if placed < num_ghosts:
        for c in range(1, width-1):
            if grid[ghost_row][c] == SPACE:
                grid[ghost_row][c] = GHOST
                placed += 1
                if placed == num_ghosts: break

    # 7) Capsules (evenly spaced on an upper-middle corridor row)
    place_capsules_square(grid, num_per_side=2)

    # 8) Fill remaining corridors with pellets, keep special tiles
    specials = {PACMAN, GHOST, CAPSULE}
    for r in range(1, height-1):
        for c in range(1, width-1):
            if grid[r][c] == SPACE:
                grid[r][c] = PELLET

    # 9) Convert to string
    return "\n".join("".join(row) for row in grid)

# Example usage:

seed_in = random.randint(0, 2**32 - 1)
seed_in
MAP_TEXT = generate_pacman_map(width=21, height=10, num_capsules=4, num_ghosts=4, symmetry="vertical", seed=seed_in)
# print(MAP_TEXT)
