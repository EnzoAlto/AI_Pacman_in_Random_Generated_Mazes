import pygame, sys, os
import random
from dataclasses import dataclass, asdict
import time, random, os
from collections import deque
from Pacman_Maze_Gen import generate_pacman_map


# -------------- TELEMETRI API for Data Logging ---------------------------
@dataclass
class GameResult:
    layout: str
    seed: int
    status: str          # "WIN" | "LOSS" | "DNF"
    score: int
    elapsed_sec: float
    moves: int
    pellets_total: int
    pellets_eaten: int
    completion_pct: float



# ---------- MAP Editor -----------------------------------
# MAP_TEXT = """
# ####################
# #...############...#
# #.#..............#.#
# #.###o########o###.#
# #........P.........#
# #.#######..#######.#
# #..................#
# #.#.#####..#####.#.#
# #.#o.G.G...G.G..o#.#
# #.#.#####..#####.#.#
# #..................#
# ####################
# """.strip("\n")

# HARD MAP. PACMAN ALWAYS GETS TRAPPED
# MAP_TEXT = """
# ############################
# #............P.............#
# #.##*###.####.###.####.###.#
# #......o............o......#
# #.##.#...########.########.#
# #.##.#......##..........##.#
# #.##.#.####.##....##.#####.#
# #.##........##.#..##.......#
# #.######..#.##.#..##.......#
# #.######..#.##.#..########.#
# #..........................#
# #.#.####.#     # #.###.###.#
# #.#..o...#.. ..#.#...o.....#ge
# #.#.####.###.###.#.###.###.#
# #..........................#
# ############################
# """.strip("\n")

# CREATE MAP. GENERATES A NEW MAP EVERY RUN
seed_in = random.randint(0, 2**32 - 1)
seed_in
GENERATE_MAP = True # Set True to Generate New Map Every Run
MAP_TEXT = generate_pacman_map(width=17, height=12, num_capsules=4, num_ghosts=4, symmetry="vertical", seed=seed_in)
# print(MAP_TEXT)


# --- Config ----------------------------------------------------------------------------------
TILE_SIZE = 30
# FPS = 60   # for testing normal pace
FPS = 120  # for rapid testing AI agents
# FPS = 180  # for Faster rapid testing AI agents
GHOST_SPRITE = 'ghost.png'  # to determine hitbox of pacman and ghosts
CAPTURE_RADIUS = TILE_SIZE * 0.45

# Tile codes
EMPTY, WALL, PELLET, POWER = 0, 1, 2, 3

# Setting Colors for each element of the game
BLACK   = (0, 0, 0)
WALL_C  = (0, 0, 255)
PELLET_C= (255, 255, 0)
POWER_C = (255,184,151)
PACMAN_C= (255, 255, 0)

# Defining How the map generator recognizes the element
CHAR2TILE = {'#': WALL, '.': PELLET, ' ': EMPTY, 'o':POWER}



# --- Map parsing -----------------------------------------------------------------
def load_map_from_text(txt):
    lines = [line.rstrip('\n') for line in txt.splitlines()]
    rows, cols = len(lines), max(len(r) for r in lines)
    lines = [r.ljust(cols, ' ') for r in lines]

    grid = [[EMPTY]*cols for _ in range(rows)]
    pac_spawn = None
    ghost_spawns = []
    #test
    for r, row in enumerate(lines):
        for c, ch in enumerate(row):
            if ch in CHAR2TILE:
                grid[r][c] = CHAR2TILE[ch]
            elif ch == 'P':
                pac_spawn = (r, c)
            elif ch == 'G':
                ghost_spawns.append((r, c))

    # If no pellets, auto-fill empties
    if not any(PELLET in row for row in grid):
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == EMPTY:
                    grid[r][c] = PELLET

    if pac_spawn is None:
        pac_spawn = (1, 1)
    if not ghost_spawns:
        ghost_spawns = [(rows-2, cols-2)]

    return grid, pac_spawn, ghost_spawns
# ---- Utilities ----------------------------------------
def manhattan(a_r, a_c, b_r, b_c): # Calculates distance between two elements (i.e. pacman or ghost)
    return abs(a_r - b_r) + abs(a_c - b_c)

def clamp(v, lo, hi): # prevent ghosts from targeting a tile outside the maze
    return max(lo, min(hi, v))

def overlap_capture(ghost, pac, radius=CAPTURE_RADIUS):
    dx = ghost.x - pac.x
    dy = ghost.y - pac.y
    return (dx*dx + dy*dy) <= (radius*radius)

# --- Utilities: Reflex Agent "AI" config ------------------------------------
AI_CONTROL = False   # set False to go back to keyboard control

REFLEX_WEIGHTS = {
    "stop_penalty": 250.0,      # discourage standing still
    "ghost_close_penalty":300.0, # penalty if a non-frightened ghost within dist <= 1. Maximum Caution to avoid Ghosts right beside pacman
    "ghost_near_penalty":150.0,   # penalty if a non-frightened ghost within dist == 2. Increases Caution for Approaching Ghosts
    "food_gain": 30,           # 1 / (1 + min_food_dist)
    "capsule_gain": 80,        # 1 / (1 + min_capsule_dist)
    "scared_ghost_gain": 10,   # 1 / (1 + min_scared_ghost_dist)
    "step_cost": 0.0,           # small negative per move if you want
    "reverse_penalty":10,

    # Corner Trap avoidance
    "corner_penalty": 0,   # L-shaped pocket
    "deadend_penalty": 0, # single exit
    "trap_scale_by_ghost_proximity": True, # scale penalty by active ghost proximity
}

#------ Utilities: Game State Loggers for AI Component ---------------------

ACTIONS = [(0,0), (0,-1), (1,0), (0,1), (-1,0)]  # Stop, Up, Right, Down, Left

def get_legal_actions(grid, r, c):
    rows, cols = len(grid), len(grid[0])
    legal = []
    for dx, dy in ACTIONS:
        nr, nc = r + dy, c + dx
        if dx == 0 and dy == 0:
            legal.append((dx,dy))  # Stop is always "legal"
        elif 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            legal.append((dx,dy))
    return legal

def successor_tile(r, c, dx, dy, grid):
    """Return tile after taking (dx,dy) from (r,c). If blocked, stay put."""
    nr, nc = r + dy, c + dx
    if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != WALL:
        return nr, nc
    return r, c

def tiles_of_type(grid, target):
    return [(rr, cc) for rr, row in enumerate(grid) for cc, v in enumerate(row) if v == target]


def ghost_legal_dirs_no_reverse(ghost, grid, allow_reverse_if_deadend=True): 
    """
    Return legal directions for this ghost.
    If not frightened, disallow the exact reverse of its last_dir,
    unless that's the *only* legal way out (dead-end) or allow_reverse_if_deadend=False.
    """
    candidates = [(0,-1),(1,0),(0,1),(-1,0)]  # U, R, D, L
    legal = [(dx,dy) for (dx,dy) in candidates if ghost._can_move(grid, dx, dy)]

    if ghost.frightened:
        return legal or [(0,0)]

    # Disallow immediate reverse (classic arcade behavior)
    rev = (-ghost.last_dir[0], -ghost.last_dir[1])
    if rev in legal and len(legal) > 1 and allow_reverse_if_deadend:
        legal.remove(rev)

    # If reverse is the only way out (dead-end), keep it so the ghost doesn't get stuck.
    return legal or [rev] or [(0,0)]

def open_neighbors_count(grid, r, c):
    rows, cols = len(grid), len(grid[0])
    opens = 0
    for dx, dy in [(0,-1),(1,0),(0,1),(-1,0)]:  # U,R,D,L
        nr, nc = r + dy, c + dx
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            opens += 1
    return opens

def is_deadend_tile(grid, r, c):
    """A tile with only 1 walkable neighbor."""
    if grid[r][c] == WALL:
        return False
    return open_neighbors_count(grid, r, c) == 1

def is_corner_tile(grid, r, c, margin=3): # marin is how many tiles to consider the reguion as a corner

    """
    Corner *region* definition: within `margin` tiles of both a vertical and a horizontal border.
    Examples (margin=2):
      - rows 0..2 with cols 0..2   -> top-left corner region
      - rows 0..2 with cols W-1-2..W-1 -> top-right
      - rows H-1-2..H-1 with cols 0..2 -> bottom-left
      - rows H-1-2..H-1 with cols W-1-2..W-1 -> bottom-right
    """
    rows, cols = len(grid), len(grid[0])
    near_top    = r <= margin
    near_bottom = r >= rows - 1 - margin
    near_left   = c <= margin
    near_right  = c >= cols - 1 - margin
    return (near_top or near_bottom) and (near_left or near_right)

# --------- MAIN PENALTY MECHANICS and ALGORITHMS -----------------------------------------
def reflex_evaluation(
        grid, pac_tile, action, ghosts, weights,
        frightened_timer=0, frightened_frames=1, # Conservative Pacman key Parameters
        history=None, breadcrumb_base=10, breadcrumb_decay=0.95, breadcrumb_k=12, # Breadcrumb memory key Parameters
        ): # Scores which legal action is best. 
    """Higher is better."""
    (pr, pc) = pac_tile
    (dx, dy) = action
    nr, nc = successor_tile(pr, pc, dx, dy, grid)

    evf = 0.0

    # Penalty. Step cost (optional shaping)
    evf -= weights["step_cost"]

    # Penalty. Stop penalty
    if (dx, dy) == (0, 0):
        evf -= weights["stop_penalty"]

    # Penalty. Food feature: closer to nearest pellet is better
    pellets = tiles_of_type(grid, PELLET)
    if pellets:
        min_food = min(manhattan(nr, nc, fr, fc) for (fr, fc) in pellets)
        evf += weights["food_gain"] * (1.0 / (1.0 + min_food))

    # Penalty. Capsule feature: encourage grabbing POWER
    capsules = tiles_of_type(grid, POWER)
    if capsules:
        min_caps = min(manhattan(nr, nc, cr, cc) for (cr, cc) in capsules) # gets closest distanec between pacman and capsules

        # rem in [0,1], where 1 = just ate power, 0 = not frightened / about to expire
        rem = (frightened_timer / float(frightened_frames)*2+1) if frightened_frames > 0 else 0
        urgency = 1.5 - rem  # grows as timer approaches 0
        # print(urgency)

        # Blend between "avoid using now" and "use soon"
        w_fright = weights.get("capsule_gain_while_frightened", 0.0)
        w_active = weights["capsule_gain"]
        cap_w = (1.0 - urgency) * w_fright + urgency * w_active
        
        # print(cap_w * (1.0 / (1.0 + min_caps)))
        evf += cap_w * (1.0 / (1.0 + min_caps))
        # evf += weights["capsule_gain"] * (1.0 / (1.0 + min_caps))

    # Penalty. Ghost proximity: avoid active ghosts, chase scared ones
    ghost_tiles = [ (int(g.y // TILE_SIZE), int(g.x // TILE_SIZE), g.frightened) for g in ghosts ]

    # Penalty. penalties for active ghosts. Behvaior of pacman to run away from ghosts
    for (gr, gc, frightened) in ghost_tiles:
        d = manhattan(nr, nc, gr, gc)
        if not frightened:
            if d <= 1:
                evf -= weights["ghost_close_penalty"]
            elif d == 2:
                evf -= weights["ghost_near_penalty"]

    # Penalty. small incentive to approach scared ghosts
    scared_positions = [(gr, gc) for (gr, gc, fr) in ghost_tiles if fr]
    if scared_positions:
        min_sg = min(manhattan(nr, nc, gr, gc) for (gr, gc) in scared_positions)
        evf += weights["scared_ghost_gain"] * (1.0 / (1.0 + min_sg))
    
    # Penalty. Breadcrumb Penalty. Encourages movement towards unexplored tiles 
    if history:
        recent = list(history)[-breadcrumb_k:]            # last K visits
        if (nr, nc) in recent:
            idx = len(recent) - 1 - recent.index((nr, nc))  # 0 = most recent
            # geometric decay: most recent gets full base, older visits decay
            evf -= breadcrumb_base * (breadcrumb_decay ** idx)


    # Penalty. Corner trap avoidance (corners/dead-ends) when ghosts are not frightened ---
    active_ghosts = [(gr, gc) for (gr, gc, fr) in ghost_tiles if not fr]
    if active_ghosts:
        # Only punish if at least one ghost is active (not frightened)
        in_deadend = is_deadend_tile(grid, nr, nc)
        # in_corner  = (not in_deadend) and is_corner_tile(grid, nr, nc)  # dead-end stronger; don't double-count
        in_corner = is_corner_tile(grid, nr, nc, margin=2)   # or corner_margin=2 for the quadrant version

        if in_deadend or in_corner:
            # Base penalty
            trap_pen = (weights.get("deadend_penalty", 120.0) if in_deadend
                        else weights.get("corner_penalty", 60.0))

            # Optionally scale by nearest active ghost distance (closer ghost => harsher penalty)
            if weights.get("trap_scale_by_ghost_proximity", True):
                min_g = min(manhattan(nr, nc, gr, gc) for (gr, gc) in active_ghosts)
                # Scale factor in (0, 1]; at distance 0/1 it's ~1, decays with distance
                scale = 1.0 / (1.0 + 0.6 * max(0, min_g - 1))
                trap_pen *= scale

            evf -= trap_pen

    # Slight nudge toward actually eating something immediately
    # (these are tiny because you'll also get score from your game logic)
    if grid[nr][nc] == PELLET:
        evf += 0.2
    elif grid[nr][nc] == POWER:
        evf += 0.5
    # print(evf)
    return evf

def choose_action_reflex(grid, pac, ghosts, weights=REFLEX_WEIGHTS,frightened_timer=0, frightened_frames=1): 
    pr, pc = pac.tile_pos()
    legal = get_legal_actions(grid, pr, pc)

    # Optional. Prevents reversing to avoid getting stuck
    # opposite = (-pac.dir_x, -pac.dir_y) if (pac.dir_x, pac.dir_y) != (0,0) else None
    # legal = [a for a in legal if a != opposite]  # uncomment to forbid immediate reversals

    # --- Reverse Score Penalty: Count + Penalty Mechanic --------------------
    scores = []
    max_reverse = 2 # How much allowable reverse before adding penalty
    cur_dir = (pac.dir_x, pac.dir_y)
    opposite = (-cur_dir[0], -cur_dir[1]) if cur_dir != (0,0) else None
    for a in legal:
        # Main Algorithm Caller
        s = reflex_evaluation(grid, (pr, pc), a, ghosts, weights,
                              frightened_timer,frightened_frames,
                              history=getattr(pac, "history", None)
                              )
        if cur_dir != (0,0) and a == opposite and pac.reverse_count >= max_reverse:
            
            excess = pac.reverse_count - max_reverse + 1
            s -= weights["reverse_penalty"]* (2 ** (excess - 1))
            # print('reverse', s) # seeing how the reverse penalty increases
        scores.append(s)


    #------ Selecting the Best Legal Move Based on scores-------------
    # scores = [reflex_evaluation(grid, (pr, pc), a, ghosts, weights) for a in legal]
    best = max(scores)
    
    # break ties randomly (like Berkeley)
    candidates = [i for i, s in enumerate(scores) if s == best]
    idx = random.choice(candidates)


    # --- Reverse Score Penalty: update reversal count ---
    if cur_dir != (0,0) and legal[idx] == opposite:
        pac.reverse_count += 1
    else:
        pac.reverse_count = 0
    
    return legal[idx]

# --- Entities: PACMAN ------------------------------------------------
class Pacman:
    def __init__(self, r, c):
        self.x = (c + 0.5) * TILE_SIZE
        self.y = (r + 0.5) * TILE_SIZE
        self.dir_x = 0
        self.dir_y = 0
        self.next_dx = 0
        self.next_dy = 0
        self.speed = 2.0  # divides TILE_SIZE (20 % 2 == 0)
        self.radius = TILE_SIZE // 2 - 2
        self._EPS_CENTER = 0.5
        self.reverse_count = 0
        self.history = deque(maxlen=12)  # recent tiles; tune length 8–20. For Breadcrumb memory

    def tile_pos(self):
        return int(self.y // TILE_SIZE), int(self.x // TILE_SIZE)

    def _center_of(self, r, c):
        return (c + 0.5) * TILE_SIZE, (r + 0.5) * TILE_SIZE

    def _is_centered(self):
        cx = (self.x % TILE_SIZE) - TILE_SIZE/2
        cy = (self.y % TILE_SIZE) - TILE_SIZE/2
        return abs(cx) <= self._EPS_CENTER and abs(cy) <= self._EPS_CENTER

    def _snap_to_center(self):
        r, c = self.tile_pos()
        self.x, self.y = self._center_of(r, c)

    def _can_move(self, grid, dx, dy):
        if dx == 0 and dy == 0:
            return True
        rows, cols = len(grid), len(grid[0])
        r, c = self.tile_pos()
        nr, nc = r + dy, c + dx
        return 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL

    def set_desired_dir(self, dx, dy, grid):
        self.next_dx, self.next_dy = dx, dy
        if self._is_centered() and self._can_move(grid, dx, dy):
            self.dir_x, self.dir_y = dx, dy

    def _advance_towards_next_center(self):
        if self.dir_x == 0 and self.dir_y == 0:
            return
        r, c = self.tile_pos()
        target_r = r + self.dir_y
        target_c = c + self.dir_x
        tx, ty = self._center_of(target_r, target_c)
        step = self.speed
        # snap if reaching center this frame
        if (tx - self.x)**2 + (ty - self.y)**2 <= step**2:
            self.x, self.y = tx, ty
            return
        if self.dir_x != 0:
            self.x = min(self.x + step, tx) if self.dir_x > 0 else max(self.x - step, tx)
            self.y = (r + 0.5) * TILE_SIZE
        else:
            self.y = min(self.y + step, ty) if self.dir_y > 0 else max(self.y - step, ty)
            self.x = (c + 0.5) * TILE_SIZE

    def update(self, grid):
        # if self._is_centered():
        #     self._snap_to_center()
        #     if (self.next_dx, self.next_dy) != (self.dir_x, self.dir_y):
        #         if self._can_move(grid, self.next_dx, self.next_dy):
        #             self.dir_x, self.dir_y = self.next_dx, self.next_dy
        #     if not self._can_move(grid, self.dir_x, self.dir_y):
        #         self.dir_x = self.dir_y = 0
        # self._advance_towards_next_center()
        # if self._is_centered():
        #     self._snap_to_center()
        #     r, c = self.tile_pos()
        #     if grid[r][c] == PELLET:
        #         grid[r][c] = EMPTY
        eaten = None
        if self._is_centered():
            self._snap_to_center()
            if (self.next_dx, self.next_dy) != (self.dir_x, self.dir_y):
                if self._can_move(grid, self.next_dx, self.next_dy):
                    self.dir_x, self.dir_y = self.next_dx, self.next_dy
            if not self._can_move(grid, self.dir_x, self.dir_y):
                self.dir_x = self.dir_y = 0
        self._advance_towards_next_center()

        # Checks if Pacman has eaten a regular pellet or a powerup
        if self._is_centered():
            self._snap_to_center()
            r, c = self.tile_pos()
            if grid[r][c] == PELLET:
                grid[r][c] = EMPTY
                eaten = 'pellet'
            elif grid[r][c] == POWER:
                grid[r][c] = EMPTY
                eaten = 'power'

            if not self.history or self.history[-1] != (r, c): # Recording the visit or tile where pellet has been consumed
                self.history.append((r, c))
        return eaten
        

    def draw(self, screen):
        pygame.draw.circle(screen, PACMAN_C, (int(self.x), int(self.y)), self.radius)

# ------ Entities: Ghosts ---------------------------------------------------------------------
class Ghost:
    def __init__(self, r, c, sprite, color=(255,0,0)):
        self.x = (c + 0.5) * TILE_SIZE
        self.y = (r + 0.5) * TILE_SIZE
        self.spawn_r, self.spawn_c = r, c
        self.dir_x = 0
        self.dir_y = 0
        self.speed = 1.5
        self.sprite = sprite
        self._EPS_CENTER = 0.5
        self.frightened = False
        self.color = color  # default red
        self.last_dir = (0, 0)   # remember last chosen direction

    def _is_centered(self):
        cx = (self.x % TILE_SIZE) - TILE_SIZE/2
        cy = (self.y % TILE_SIZE) - TILE_SIZE/2
        return abs(cx) <= self._EPS_CENTER and abs(cy) <= self._EPS_CENTER

    def _snap_to_center(self):
        r, c = int(self.y // TILE_SIZE), int(self.x // TILE_SIZE)
        self.x = (c + 0.5) * TILE_SIZE
        self.y = (r + 0.5) * TILE_SIZE

    def _can_move(self, grid, dx, dy):
        rows, cols = len(grid), len(grid[0])
        r, c = int(self.y // TILE_SIZE), int(self.x // TILE_SIZE)
        nr, nc = r + dy, c + dx
        return 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL

    def _choose_towards_target(self, grid, tgt_r, tgt_c):
        row = int(self.y // TILE_SIZE)
        col = int(self.x // TILE_SIZE)
        best_d, best = 1e9, (0, 0)

        for dx, dy in ghost_legal_dirs_no_reverse(self, grid):
            nr, nc = row + dy, col + dx
            d = abs(nr - tgt_r) + abs(nc - tgt_c)
            if d < best_d:
                best_d, best = d, (dx, dy)

        self.dir_x, self.dir_y = best
        self.last_dir = best  # commit chosen direction
        

    def _choose_dir_away(self, grid, pac_r, pac_c):
        row = int(self.y // TILE_SIZE)
        col = int(self.x // TILE_SIZE)
        best_d, best = -1, (0, 0)

        for dx, dy in ghost_legal_dirs_no_reverse(self, grid):
            nr, nc = row + dy, col + dx
            d = abs(nr - pac_r) + abs(nc - pac_c)
            if d > best_d:
                best_d, best = d, (dx, dy)

        self.dir_x, self.dir_y = best
        self.last_dir = best  # commit chosen direction

    def _advance_towards_next_center(self):
        if self.dir_x == 0 and self.dir_y == 0:
            return
        r, c = int(self.y // TILE_SIZE), int(self.x // TILE_SIZE)
        tx = (c + 0.5 + self.dir_x) * TILE_SIZE
        ty = (r + 0.5 + self.dir_y) * TILE_SIZE
        step = self.speed
        if (tx - self.x)**2 + (ty - self.y)**2 <= step**2:
            self.x, self.y = tx, ty
            return
        if self.dir_x != 0:
            self.x = min(self.x + step, tx) if self.dir_x > 0 else max(self.x - step, tx)
            self.y = (r + 0.5) * TILE_SIZE
        else:
            self.y = min(self.y + step, ty) if self.dir_y > 0 else max(self.y - step, ty)
            self.x = (c + 0.5) * TILE_SIZE

    # ---- MAIN Ghost Update (calls strategy) ---------------------------------------------
    def update(self, grid, pac_r, pac_c, pac_dx, pac_dy, blinky=None, rows_cols=None):
        if self._is_centered():
            self._snap_to_center()
            if self.frightened:
                self._choose_dir_away(grid, pac_r, pac_c)
                self.speed = 1.5
            else:
                tr, tc = self.target_tile(grid, pac_r, pac_c, pac_dx, pac_dy, blinky, rows_cols) # Calling the Chase Strategy dependend on ghost color
                self._choose_towards_target(grid, tr, tc)
                self.speed = 2.0
            if not self._can_move(grid, self.dir_x, self.dir_y):
                self.dir_x = self.dir_y = 0
        self._advance_towards_next_center()

    def respawn_to_spawn(self):
        self.x = (self.spawn_c + 0.5) * TILE_SIZE
        self.y = (self.spawn_r + 0.5) * TILE_SIZE
        self.dir_x = self.dir_y = 0

    def draw(self, screen):
        # frightened = blue; otherwise use ghost’s own color
        if self.frightened or not (self.sprite and hasattr(self.sprite, "get_rect")):
            color = (100,100,255) if self.frightened else self.color
            pygame.draw.circle(screen, color, (int(self.x), int(self.y)), TILE_SIZE//2 - 2)
        else:
            rect = self.sprite.get_rect(center=(int(self.x), int(self.y)))
            screen.blit(self.sprite, rect)

    # Default strategy (Blinky-like); subclasses override this.
    def target_tile(self, grid, pac_r, pac_c, pac_dx, pac_dy, blinky, rows_cols):
        return pac_r, pac_c

# ------------- ENTITIES: Ghost Strategies -=--------------------------------------------
# Pinky – ambusher (4 ahead, UP bug = 4 up + 4 left)
class Pinky(Ghost):
    def __init__(self, r, c, sprite):
        super().__init__(r, c, sprite, color=(255,192,203))  # pink
    def target_tile(self, grid, pac_r, pac_c, pac_dx, pac_dy, blinky, rows_cols):
        rows, cols = rows_cols
        # Classic bug: when Pac-Man faces UP, target = 4 up + 4 left
        if (pac_dx, pac_dy) == (0, -1):
            tr = pac_r - 4
            tc = pac_c - 4
        else:
            tr = pac_r + 4 * pac_dy
            tc = pac_c + 4 * pac_dx
        tr = clamp(tr, 0, rows-1); tc = clamp(tc, 0, cols-1)
        return tr, tc

# Inky – vector trick: Blinky (2 ahead of Pac-Man), then double the vector
class Inky(Ghost):
    def __init__(self, r, c, sprite):
        super().__init__(r, c, sprite, color=(0,255,255))  # cyan
    def target_tile(self, grid, pac_r, pac_c, pac_dx, pac_dy, blinky, rows_cols):
        rows, cols = rows_cols
        # 1) point two tiles ahead of Pac-Man
        ahead_r = pac_r + 2 * pac_dy
        ahead_c = pac_c + 2 * pac_dx
        ahead_r = clamp(ahead_r, 0, rows-1); ahead_c = clamp(ahead_c, 0, cols-1)
        # 2) vector from Blinky to that point (needs blinky position!)
        if blinky is None:
            return ahead_r, ahead_c
        br, bc = int(blinky.y // TILE_SIZE), int(blinky.x // TILE_SIZE)
        vr, vc = (ahead_r - br), (ahead_c - bc)
        # 3) double the vector
        tr, tc = (br + 2*vr), (bc + 2*vc)
        tr = clamp(tr, 0, rows-1); tc = clamp(tc, 0, cols-1)
        return tr, tc

# Clyde – chase if far; else scatter to bottom-left corner
class Clyde(Ghost):
    def __init__(self, r, c, sprite, seed=None, epsilon=0.9, wander_distance=6):
        """
        epsilon: probability (0–1) of Clyde wandering randomly instead of normal logic
        wander_distance: radius around Pacman where Clyde may wander when random
        """
        super().__init__(r, c, sprite, color=(255,165,0))  # orange
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.wander_distance = wander_distance

    def _random_near_pacman(self, grid, pac_r, pac_c):
        """Pick a random walkable tile within wander_distance of Pac-Man"""
        walkable = []
        for rr in range(max(0, pac_r - self.wander_distance), min(len(grid), pac_r + self.wander_distance + 1)):
            for cc in range(max(0, pac_c - self.wander_distance), min(len(grid[0]), pac_c + self.wander_distance + 1)):
                if grid[rr][cc] == 0:  # adjust if 0 means walkable
                    walkable.append((rr, cc))
        return self.rng.choice(walkable) if walkable else (pac_r, pac_c)

    def target_tile(self, grid, pac_r, pac_c, pac_dx, pac_dy, blinky, rows_cols):
        rows, cols = rows_cols
        my_r, my_c = int(self.y // TILE_SIZE), int(self.x // TILE_SIZE)

        # With probability epsilon, Clyde "wanders"
        if self.rng.random() < self.epsilon:
            return self._random_near_pacman(grid, pac_r, pac_c)

        # Normal Clyde behavior
        d = manhattan(my_r, my_c, pac_r, pac_c)
        if d > 8:
            return pac_r, pac_c  # chase like Blinky
        return rows - 2, 1  # scatter corner





# --- Setup & main loop: MAP GENERATION --------------------------------------------------------------------------------------------------
def make_game_from_text(txt):
    grid, pac_spawn, ghost_spawns = load_map_from_text(txt)
    rows, cols = len(grid), len(grid[0])
    width, height = cols*TILE_SIZE, rows*TILE_SIZE

    pac = Pacman(*pac_spawn)
    ghost_sprite = None
    if os.path.exists(GHOST_SPRITE):
        try:
            ghost_sprite = pygame.image.load(GHOST_SPRITE).convert_alpha()
            ghost_sprite = pygame.transform.scale(ghost_sprite, (TILE_SIZE, TILE_SIZE))
        except Exception:
            ghost_sprite = None

    ghosts = []
    for i, (r, c) in enumerate(ghost_spawns):
        if i == 0:
            ghosts.append(Ghost(r, c, ghost_sprite, color=(255,0,0)))         # Blinky (red)
        elif i == 1:
            ghosts.append(Pinky(r, c, ghost_sprite))                           # Pinky
        elif i == 2:
            ghosts.append(Inky(r, c, ghost_sprite))                            # Inky
        elif i == 3:
            ghosts.append(Clyde(r, c, ghost_sprite))                           # Clyde
        else:
            ghosts.append(Ghost(r, c, ghost_sprite, color=(255,0,0)))         # extra Blinkies if more spawns

    return grid, pac, ghosts, rows, cols, width, height

def count_pellets(g):
    return sum(cell == PELLET for row in g for cell in row)


# --- Setup & main loop: API CALLING --------------------------------------------------------------------------------------------------
def run_single_game_telemetry(
    layout_name: str = "inline",
    seed: int | None = None,
    max_time_sec: float | None = None,
    max_moves: int | None = None,
    headless: bool = True
) -> dict:
    global MAP_TEXT, GENERATE_MAP, AI_CONTROL

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    # seed = random.seed(seed)
    seed = random.randint(0, 10000)
    # print(seed)

    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    if GENERATE_MAP:
        seed_in = random.randint(0, 2**32 - 1)
        MAP_TEXT = generate_pacman_map(width=17, height=12, num_capsules=4, num_ghosts=4, symmetry="vertical", seed=seed_in)
        
    grid, pac, ghosts, ROWS, COLS, WIDTH, HEIGHT = make_game_from_text(MAP_TEXT)

    screen = None
    font = None
    if not headless:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pac-Man (batch)")
        font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()

    pellets_left = count_pellets(grid)
    pellets_total = int(pellets_left)
    pellets_eaten = 0
    score = 0
    ghost_chain = 0

    FRIGHTENED_SECS = 6
    FRIGHTENED_FRAMES = int(FPS * FRIGHTENED_SECS)
    frightened_timer = 0

    start_time = time.perf_counter()
    moves = 0
    status = None

    running = True
    while running:
        clock.tick(FPS)
        moves += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                status = "DNF"
                running = False

        if AI_CONTROL and pac._is_centered():
            dx, dy = choose_action_reflex(grid, pac, ghosts, REFLEX_WEIGHTS, frightened_timer, FRIGHTENED_FRAMES)
            pac.set_desired_dir(dx, dy, grid)

        eaten = pac.update(grid)
        if eaten == 'pellet':
            score += 10
            pellets_left -= 1
            pellets_eaten += 1
        elif eaten == 'power':
            score += 50
            frightened_timer = FRIGHTENED_FRAMES
            for g in ghosts:
                g.frightened = True

        pr, pc = pac.tile_pos()
        rows_cols = (ROWS, COLS)
        blinky = ghosts[0] if ghosts else None
        pac_dx, pac_dy = pac.dir_x, pac.dir_y

        if frightened_timer > 0:
            frightened_timer -= 1
            if frightened_timer == 0:
                for g in ghosts:
                    g.frightened = False

        for g in ghosts:
            g.update(grid, pr, pc, pac_dx, pac_dy, blinky=blinky, rows_cols=rows_cols)
            # gr, gc = int(g.y // TILE_SIZE), int(g.x // TILE_SIZE)
            # if (gr, gc) == (pr, pc):
            if overlap_capture(g, pac):
                if g.frightened:
                    ghost_chain += 1
                    score += 200 * (2 ** (ghost_chain - 1))
                    g.respawn_to_spawn()
                    g.frightened = False
                else:
                    status = "LOSS"
                    running = False
                    break

        if status is None and pellets_left <= 0:
            status = "WIN"
            running = False

        elapsed = time.perf_counter() - start_time
        if status is None and (
            (max_time_sec is not None and elapsed >= max_time_sec) or
            (max_moves is not None and moves >= max_moves)
        ):
            status = "DNF"
            running = False

        # ---- RENDER (only when not headless) ----
        if not headless and screen is not None:
            screen.fill(BLACK)
            # tiles
            for r in range(ROWS):
                for c in range(COLS):
                    cell = grid[r][c]
                    if cell == WALL:
                        pygame.draw.rect(screen, WALL_C, (c*TILE_SIZE, r*TILE_SIZE, TILE_SIZE, TILE_SIZE))
                    elif cell == PELLET:
                        pygame.draw.circle(screen, PELLET_C, (c*TILE_SIZE + TILE_SIZE//2, r*TILE_SIZE + TILE_SIZE//2), 3)
                    elif cell == POWER:
                        pygame.draw.circle(screen, POWER_C, (c*TILE_SIZE + TILE_SIZE//2, r*TILE_SIZE + TILE_SIZE//2), 7)
            # actors
            pac.draw(screen)
            for g in ghosts:
                g.draw(screen)
            # HUD (optional)
            if font is not None:
                txt = font.render(f"Score: {score}  Pellets left: {pellets_left}", True, (200,200,200))
                screen.blit(txt, (6, 4))
            pygame.display.flip()

    elapsed_sec = time.perf_counter() - start_time
    completion_pct = (pellets_eaten / pellets_total * 100.0) if pellets_total else 0.0

    # ---- CLEANUP (both modes) ----
    if screen is not None:
        pygame.display.quit()
    pygame.quit()

    return asdict(GameResult(
        layout=layout_name,
        seed=seed,
        status=status or "DNF",
        score=int(score),
        elapsed_sec=round(elapsed_sec, 6),
        moves=int(moves),
        pellets_total=int(pellets_total),
        pellets_eaten=int(pellets_eaten),
        completion_pct=round(completion_pct, 2),
    ))


# --- Setup & main loop: MAIN LOOP RUN --------------------------------------------------------------------------------------------------

def main():
    pygame.init()
    global AI_CONTROL
    grid, pac, ghosts, ROWS, COLS, WIDTH, HEIGHT = make_game_from_text(MAP_TEXT)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pac-Man (inline map, simple)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 18)

    pellets_left = count_pellets(grid)

    # -----SCORING COUNTER -----------------------------------
    score = 0
    ghost_chain = 0

    # --- NEW: frightened state ---
    FRIGHTENED_SECS = 6
    FRIGHTENED_FRAMES = int(FPS * FRIGHTENED_SECS)
    frightened_timer = 0

    def reset_game():
        nonlocal grid, pac, ghosts, pellets_left, ROWS, COLS, WIDTH, HEIGHT, frightened_timer, ghost_chain, score
        global MAP_TEXT, GENERATE_MAP,AI_CONTROL

        if GENERATE_MAP:
            seed_in = random.randint(0, 2**32 - 1)
            MAP_TEXT = generate_pacman_map(width=17, height=12, num_capsules=4, num_ghosts=4, symmetry="vertical", seed=seed_in)
        
        grid, pac_spawn, ghost_spawns = load_map_from_text(MAP_TEXT)
        pac = Pacman(*pac_spawn)
        spr = ghosts[0].sprite if ghosts else None

        ghosts = []
        for i, (r, c) in enumerate(ghost_spawns):
            if i == 0:
                ghosts.append(Ghost(r, c, spr, color=(255,0,0)))
            elif i == 1:
                ghosts.append(Pinky(r, c, spr))
            elif i == 2:
                ghosts.append(Inky(r, c, spr))
            elif i == 3:
                ghosts.append(Clyde(r, c, spr))
            else:
                ghosts.append(Ghost(r, c, spr, color=(255,0,0)))

        ROWS, COLS = len(grid), len(grid[0])
        WIDTH, HEIGHT = COLS*TILE_SIZE, ROWS*TILE_SIZE
        pellets_left = count_pellets(grid)
        frightened_timer = 0 # reset timer
        ghost_chain = 0 # Reset ghost chain score
        score = 0

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:    pac.set_desired_dir(0, -1, grid)
                if event.key == pygame.K_DOWN:  pac.set_desired_dir(0,  1, grid)
                if event.key == pygame.K_LEFT:  pac.set_desired_dir(-1, 0, grid)
                if event.key == pygame.K_RIGHT: pac.set_desired_dir(1,  0, grid)
                if event.key == pygame.K_r:     reset_game()

        # ------ AI control. Overwrite Key movement commands based on AI Decisions -------------------
        if AI_CONTROL:
            if pac._is_centered():
                dx, dy = choose_action_reflex(grid, pac, ghosts, REFLEX_WEIGHTS, frightened_timer, FRIGHTENED_FRAMES)
                pac.set_desired_dir(dx, dy, grid)

        # Pac-Man update + consume mechanic 
        eaten = pac.update(grid)
        if eaten == 'pellet':
            score += 10
        elif eaten == 'power':
            score += 50
            frightened_timer = FRIGHTENED_FRAMES
            # ghost_chain = 0                 # start a fresh combo window. Unhide to let scoring reset if another Power is eaten
            for g in ghosts:
                g.frightened = True

        # Collecting Pacman Data to update Ghost Strategy 
        pr, pc = pac.tile_pos() # Colelct Pacman's position
        rows_cols = (ROWS, COLS)
        blinky = ghosts[0] if ghosts else None # Allows Inky's Strategy to target a tile ahead of pacman dependent on the distance of pacman and blinky
        pac_dx, pac_dy = pac.dir_x, pac.dir_y # Collects direction of pacman's movement, this allows Pinky and Inky to decide their chase strategy to pacman

        # Decrement frightened timer 
        if frightened_timer > 0:
            frightened_timer -= 1
            if frightened_timer == 0:
                for g in ghosts:
                    g.frightened = False


        # Ghosts update + collisions mechanic
        for g in ghosts:
            g.update(grid, pr, pc, pac_dx, pac_dy, blinky=blinky, rows_cols=rows_cols)
            # gr, gc = int(g.y//TILE_SIZE), int(g.x//TILE_SIZE)
            # if (gr, gc) == (pr, pc):
            if overlap_capture(g, pac):
                if g.frightened:
                    ghost_chain += 1
                    points = 200 * (2 ** (ghost_chain - 1)) # score for eating ghosts doubles 
                    score += points
                    g.respawn_to_spawn()
                    g.frightened = False
                    continue
                else:
                    reset_game()
                    break

        # Win check (unchanged)
        pellets_left = count_pellets(grid)
        if pellets_left == 0:
            for r in range(ROWS):
                for c in range(COLS):
                    if grid[r][c] == EMPTY:
                        grid[r][c] = PELLET
            pellets_left = count_pellets(grid)
            reset_game()

        # Draw
        screen.fill(BLACK)
        for r in range(ROWS):
            for c in range(COLS):
                cell = grid[r][c]
                if cell == WALL:
                    pygame.draw.rect(screen, WALL_C, (c*TILE_SIZE, r*TILE_SIZE, TILE_SIZE, TILE_SIZE))
                elif cell == PELLET:
                    pygame.draw.circle(screen, PELLET_C, (c*TILE_SIZE + TILE_SIZE//2, r*TILE_SIZE + TILE_SIZE//2), 3)
                elif cell == POWER:
                    pygame.draw.circle(screen, POWER_C, (c*TILE_SIZE + TILE_SIZE//2, r*TILE_SIZE + TILE_SIZE//2), 7)

        pac.draw(screen)
        for g in ghosts:
            g.draw(screen)

        frightened_any = any(g.frightened for g in ghosts)

        text = font.render(
            # f"Pellets: {pellets_left}    Power: {'ON' if frightened_any else 'off'}    [Arrows] Move  [R] Reset",
            f"Score: {score}    Power: {'ON' if frightened_any else 'off'}    [Arrows] Move  [R] Reset",

            True, (200,200,200)
        )
        screen.blit(text, (6, 4))
        pygame.display.flip()

    pygame.quit(); sys.exit()


if __name__ == "__main__":
    main()
