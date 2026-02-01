from collections import deque
from typing import Tuple, Optional, List, Dict, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from game_constants import FoodType, ShopCosts, GameConstants, Team
from robot_controller import RobotController
from item import Pan, Plate, Food


class RouteCalculator:
    """
    Reads the map and maintains path reservations with a per-round timer.
    - If a robot reserves a path, those cells are marked until the robot advances (one step released per round).
    - Another robot that wants to walk searches against this record; if the path is not possible it should give up.
    - If possible, both robots can move (their paths don't conflict).
    """

    # Neighbors for Chebyshev distance 1 (8-direction)
    _NEIGHBORS = [(dx, dy) for dx in (-1, 0, 1)
                  for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

    def __init__(self, width: int, height: int, is_walkable: Callable[[int, int], bool]):
        self._width = width
        self._height = height
        self._is_walkable = is_walkable
        # bot_id -> {"path": [(x,y), ...], "step": int}. step = current index (0 = at start).
        self._reservations: dict[int, dict] = {}

    def update_map(self, width: int, height: int, is_walkable: Callable[[int, int], bool]) -> None:
        """Refresh map dimensions and walkability (e.g. each turn from controller.get_map(team))."""
        self._width = width
        self._height = height
        self._is_walkable = is_walkable

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._width and 0 <= y < self._height

    def get_reserved_cells(self, exclude_bot_id: Optional[int] = None) -> Set[Tuple[int, int]]:
        """Cells currently reserved by some robot (from current position to end of path)."""
        out: Set[Tuple[int, int]] = set()
        for bid, data in self._reservations.items():
            if exclude_bot_id is not None and bid == exclude_bot_id:
                continue
            path = data["path"]
            step = data["step"]
            for i in range(step, len(path)):
                out.add(path[i])
        return out

    def find_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        blocked_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        BFS from start towards end. Goal is to reach any walkable cell from which
        the bot can touch end (i.e. on or adjacent to end). The end cell does not
        need to be walkable. A cell is passable if walkable and not in blocked_cells.
        Returns full path [start, ..., cell_that_touches_end] or None if no path.
        """
        if blocked_cells is None:
            blocked_cells = set()
        sx, sy = start
        ex, ey = end
        if not self._in_bounds(sx, sy) or not self._in_bounds(ex, ey):
            return None
        if not self._is_walkable(sx, sy):
            return None
        # Already at or adjacent to end (can touch it)
        if max(abs(sx - ex), abs(sy - ey)) <= 1:
            return [start]

        queue: deque[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = deque([
                                                                            (start, [start])])
        visited: Set[Tuple[int, int]] = {start}

        while queue:
            (x, y), path = queue.popleft()
            for dx, dy in self._NEIGHBORS:
                nx, ny = x + dx, y + dy
                if not self._in_bounds(nx, ny) or (nx, ny) in visited:
                    continue
                if not self._is_walkable(nx, ny) or (nx, ny) in blocked_cells:
                    continue
                visited.add((nx, ny))
                new_path = path + [(nx, ny)]
                # Reached a cell from which we can touch end (on or adjacent)
                if max(abs(nx - ex), abs(ny - ey)) <= 1:
                    return new_path
                queue.append(((nx, ny), new_path))
        return None

    def can_reach(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        exclude_bot_id: Optional[int] = None,
        extra_blocked_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
        """
        Check if a path exists from start to end given current reservations
        and extra blocked cells (e.g. other robots' positions).
        exclude_bot_id: when computing for one robot, ignore its own reservation.
        Returns (True, path) or (False, None).
        """
        blocked = self.get_reserved_cells(exclude_bot_id=exclude_bot_id)
        if extra_blocked_cells:
            blocked = blocked | extra_blocked_cells
        path = self.find_path(start, end, blocked_cells=blocked)
        return (path is not None, path)

    def try_reserve_path(
        self,
        bot_id: int,
        start: Tuple[int, int],
        end: Tuple[int, int],
        extra_blocked_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
        """
        Find a path from start to end considering other robots' reservations
        and extra blocked cells (e.g. other robots' positions).
        If a path exists, reserve it for bot_id and return (True, path).
        If not accessible, return (False, None) — caller should give up.
        """
        ok, path = self.can_reach(
            start, end, exclude_bot_id=bot_id, extra_blocked_cells=extra_blocked_cells
        )
        if not ok or path is None:
            return (False, None)
        self._reservations[bot_id] = {"path": path, "step": 0, "target": end}
        return (True, path)

    def reserve_path(self, bot_id: int, path: List[Tuple[int, int]]) -> bool:
        """
        Manually reserve a path for bot_id (e.g. one you computed yourself).
        path must be a valid sequence of adjacent cells. Returns True if stored.
        """
        if not path:
            return False
        self._reservations[bot_id] = {"path": list(
            path), "step": 0, "target": path[-1]}
        return True

    def advance_round(self) -> None:
        """
        Call once per game round. Each reservation advances by one step;
        when a robot has passed the last cell, its reservation is removed.
        """
        to_remove: List[int] = []
        for bot_id, data in self._reservations.items():
            data["step"] += 1
            if data["step"] >= len(data["path"]):
                to_remove.append(bot_id)
        for bot_id in to_remove:
            del self._reservations[bot_id]

    def get_next_step(self, bot_id: int) -> Optional[Tuple[int, int]]:
        """
        For a bot with a reserved path, return (dx, dy) to move one step along the path this round.
        Returns None if no reservation or already at end.
        """
        data = self._reservations.get(bot_id)
        if data is None:
            return None
        path = data["path"]
        step = data["step"]
        if step + 1 >= len(path):
            return None
        x0, y0 = path[step]
        x1, y1 = path[step + 1]
        return (x1 - x0, y1 - y0)

    def release_path(self, bot_id: int) -> None:
        """Remove reservation for bot_id (e.g. when giving up)."""
        self._reservations.pop(bot_id, None)

    def has_reservation(
        self, bot_id: int, target: Optional[Tuple[int, int]] = None
    ) -> bool:
        """True if bot has a reservation; if target is given, also requires it matches the reserved target."""
        if bot_id not in self._reservations:
            return False
        if target is None:
            return True
        return self._reservations[bot_id].get("target") == target

    def is_cell_reserved(self, x: int, y: int, exclude_bot_id: Optional[int] = None) -> bool:
        return (x, y) in self.get_reserved_cells(exclude_bot_id=exclude_bot_id)


# ---------------------------------------------------------------------------
# ResourceManager: discover resources, track status, apply/release for exclusive use.
# Call update() once per turn (before bots act) to refresh status and auto-release.
# ---------------------------------------------------------------------------

class CounterStatus(Enum):
    """Status of a counter tile."""
    EMPTY = "empty"           # No item on counter
    HAS_FOOD = "has_food"     # Raw food (for chop)
    HAS_PLATE = "has_plate"   # Plate (possibly with food)
    HAS_OTHER = "has_other"   # Other items (e.g. pan)


class CookerStatus(Enum):
    """Status of a cooker tile (based on pan and food state)."""
    NO_PAN = "no_pan"         # Pan missing (accidentally removed)
    PAN_EMPTY = "pan_empty"   # Pan present, no food
    RAW_FOOD = "raw_food"     # Raw food in pan, cooking
    COOKED = "cooked"         # Food ready to take (cooked_stage == 1)
    BURNT = "burnt"           # Food burnt (cooked_stage == 2)


def _chebyshev_dist(ax: int, ay: int, bx: int, by: int) -> int:
    """Chebyshev (king move) distance."""
    return max(abs(ax - bx), abs(ay - by))


# 8-direction neighbors (Chebyshev / king move)
_BFS_NEIGHBORS = [(dx, dy) for dx in (-1, 0, 1)
                  for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


class ResourceManager:
    """
    Manages cookers and counters: discovers locations, tracks status, and provides
    apply/release semantics so bots request exclusive use before interacting.
    - apply_* returns closest available resource (if applier_location given) or first available
    - Optional time_limit: auto-release after that many turns
    """

    def __init__(self):
        # (x, y) -> position of each resource
        self._cookers: List[Tuple[int, int]] = []
        self._counters: List[Tuple[int, int]] = []
        # Positions that are held (no owner tracked)
        self._cooker_held: Set[Tuple[int, int]] = set()
        self._counter_held: Set[Tuple[int, int]] = set()
        # (x, y) -> {"applied_turn": int, "time_limit": Optional[int]}
        self._counter_hold_meta: Dict[Tuple[int, int], dict] = {}
        self._cooker_hold_meta: Dict[Tuple[int, int], dict] = {}
        # Cached status (updated each turn via update())
        self._cooker_status: Dict[Tuple[int, int], CookerStatus] = {}
        self._counter_status: Dict[Tuple[int, int], CounterStatus] = {}
        self._current_turn: int = 0
        # Map for BFS (no robot blocking) - updated each turn
        self._width: int = 0
        self._height: int = 0
        self._is_walkable: Optional[Callable[[int, int], bool]] = None

    def discover_resources(self, controller: RobotController) -> None:
        """
        Scan the map for all cookers and counters. Call once at the beginning.
        """
        self._cookers.clear()
        self._counters.clear()
        self._cooker_held.clear()
        self._counter_held.clear()
        self._counter_hold_meta.clear()
        self._cooker_hold_meta.clear()
        self._cooker_status.clear()
        self._counter_status.clear()

        m = controller.get_map(controller.get_team())
        self._width = m.width
        self._height = m.height
        self._is_walkable = m.is_tile_walkable
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == "COOKER":
                    self._cookers.append((x, y))
                elif tile.tile_name == "COUNTER":
                    self._counters.append((x, y))

    def update(self, controller: RobotController) -> None:
        """
        Call once per game turn (before bots act). Refreshes status from the actual
        tile state, auto-releases when time limit is reached, and auto-releases
        cookers when burnt or pan is missing. Also updates map for BFS accessibility.
        """
        self._current_turn = controller.get_turn()
        team = controller.get_team()
        m = controller.get_map(team)
        self._width = m.width
        self._height = m.height
        self._is_walkable = m.is_tile_walkable
        for pos in self._cookers:
            self._update_cooker_status(controller, team, pos)
            self._maybe_auto_release_cooker_by_time(pos)
            self._maybe_auto_release_cooker(pos)
        for pos in self._counters:
            self._update_counter_status(controller, team, pos)
            self._maybe_auto_release_counter_by_time(pos)

    def _update_cooker_status(self, controller: RobotController, team: Team, pos: Tuple[int, int]) -> None:
        tile = controller.get_tile(team, pos[0], pos[1])
        if tile is None:
            self._cooker_status[pos] = CookerStatus.NO_PAN
            return
        item = getattr(tile, "item", None)
        if not isinstance(item, Pan):
            self._cooker_status[pos] = CookerStatus.NO_PAN
            return
        if item.food is None:
            self._cooker_status[pos] = CookerStatus.PAN_EMPTY
            return
        food = item.food
        if food.cooked_stage == 2:
            self._cooker_status[pos] = CookerStatus.BURNT
        elif food.cooked_stage == 1:
            self._cooker_status[pos] = CookerStatus.COOKED
        else:
            self._cooker_status[pos] = CookerStatus.RAW_FOOD

    def _update_counter_status(self, controller: RobotController, team: Team, pos: Tuple[int, int]) -> None:
        tile = controller.get_tile(team, pos[0], pos[1])
        if tile is None:
            self._counter_status[pos] = CounterStatus.EMPTY
            return
        item = getattr(tile, "item", None)
        if item is None:
            self._counter_status[pos] = CounterStatus.EMPTY
            return
        if isinstance(item, Food):
            self._counter_status[pos] = CounterStatus.HAS_FOOD
        elif isinstance(item, Plate):
            self._counter_status[pos] = CounterStatus.HAS_PLATE
        else:
            self._counter_status[pos] = CounterStatus.HAS_OTHER

    def _maybe_auto_release_cooker(self, pos: Tuple[int, int]) -> None:
        """Auto-release cooker when burnt or pan is missing."""
        status = self._cooker_status.get(pos, CookerStatus.NO_PAN)
        if status in (CookerStatus.BURNT, CookerStatus.NO_PAN):
            self._cooker_held.discard(pos)
            self._cooker_hold_meta.pop(pos, None)

    def _maybe_auto_release_cooker_by_time(self, pos: Tuple[int, int]) -> None:
        """Auto-release cooker when time limit is reached."""
        meta = self._cooker_hold_meta.get(pos)
        if meta is None:
            return
        limit = meta.get("time_limit")
        if limit is None:
            return
        if (self._current_turn - meta["applied_turn"]) >= limit:
            self._cooker_held.discard(pos)
            self._cooker_hold_meta.pop(pos, None)

    def _maybe_auto_release_counter_by_time(self, pos: Tuple[int, int]) -> None:
        """Auto-release counter when time limit is reached."""
        meta = self._counter_hold_meta.get(pos)
        if meta is None:
            return
        limit = meta.get("time_limit")
        if limit is None:
            return
        if (self._current_turn - meta["applied_turn"]) >= limit:
            self._counter_held.discard(pos)
            self._counter_hold_meta.pop(pos, None)

    def _bfs_distance_to_resource(
        self, from_pos: Tuple[int, int], resource_pos: Tuple[int, int]
    ) -> Optional[int]:
        """
        BFS shortest path length from from_pos to any cell adjacent to resource_pos.
        Uses map walkability only - no robot blocking.
        Returns None if resource is not accessible.
        """
        if self._is_walkable is None:
            return None  # Map not yet available, cannot verify accessibility
        sx, sy = from_pos
        rx, ry = resource_pos
        if not (0 <= sx < self._width and 0 <= sy < self._height):
            return None
        if not self._is_walkable(sx, sy):
            return None
        goals: Set[Tuple[int, int]] = set()
        for dx, dy in _BFS_NEIGHBORS:
            ax, ay = rx + dx, ry + dy
            if 0 <= ax < self._width and 0 <= ay < self._height and self._is_walkable(ax, ay):
                goals.add((ax, ay))
        if not goals:
            return None
        if (sx, sy) in goals:
            return 0
        queue: deque[Tuple[int, int, int]] = deque([(sx, sy, 0)])
        visited: Set[Tuple[int, int]] = {(sx, sy)}
        while queue:
            x, y, d = queue.popleft()
            for dx, dy in _BFS_NEIGHBORS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self._width and 0 <= ny < self._height):
                    continue
                if (nx, ny) in visited:
                    continue
                if not self._is_walkable(nx, ny):
                    continue
                visited.add((nx, ny))
                new_d = d + 1
                if (nx, ny) in goals:
                    return new_d
                queue.append((nx, ny, new_d))
        return None

    def _find_closest_available(
        self,
        positions: List[Tuple[int, int]],
        held: Set[Tuple[int, int]],
        from_pos: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """
        Among available (not held) positions, return closest by BFS shortest path.
        Only includes resources accessible from from_pos. If from_pos is None, returns first available.
        """
        available = [p for p in positions if p not in held]
        if not available:
            return None
        if from_pos is None:
            return available[0]
        accessible: List[Tuple[Tuple[int, int], int]] = []
        for p in available:
            dist = self._bfs_distance_to_resource(from_pos, p)
            if dist is not None:
                accessible.append((p, dist))
        if not accessible:
            return None
        return min(accessible, key=lambda x: x[1])[0]

    # ---------- Counter ----------

    def apply_counter(
        self,
        applier_location: Optional[Tuple[int, int]] = None,
        time_limit: Optional[int] = None,
        specific_pos: Optional[Tuple[int, int]] = None,
    ) -> Tuple[bool, Optional[Tuple[int, int]], CounterStatus]:
        """
        Request exclusive use of a counter. Resource is held (no owner tracked).
        - applier_location: (x, y); if given, returns closest available counter
        - time_limit: turns until auto-release (None = no limit)
        - specific_pos: if given, request that counter only (ignores applier_location)
        Returns (success, pos, status). If no resource available: (False, None, EMPTY).
        """
        if specific_pos is not None:
            if specific_pos not in self._counters or specific_pos in self._counter_held:
                candidates = []
            elif applier_location is not None and self._bfs_distance_to_resource(applier_location, specific_pos) is None:
                candidates = []  # not accessible from applier
            else:
                candidates = [specific_pos]
        else:
            pos = self._find_closest_available(
                self._counters, self._counter_held, applier_location)
            candidates = [pos] if pos is not None else []
        if not candidates:
            return (False, None, CounterStatus.EMPTY)
        pos = candidates[0]
        if pos in self._counter_held:
            return (False, None, self._counter_status.get(pos, CounterStatus.EMPTY))
        self._counter_held.add(pos)
        self._counter_hold_meta[pos] = {
            "applied_turn": self._current_turn, "time_limit": time_limit}
        return (True, pos, self._counter_status.get(pos, CounterStatus.EMPTY))

    def release_counter(self, pos: Tuple[int, int]) -> bool:
        """Release the counter. Returns True if it was held."""
        if pos in self._counter_held:
            self._counter_held.discard(pos)
            self._counter_hold_meta.pop(pos, None)
            return True
        return False

    def get_counter_status(self, pos: Tuple[int, int]) -> CounterStatus:
        return self._counter_status.get(pos, CounterStatus.EMPTY)

    def is_counter_held(self, pos: Tuple[int, int]) -> bool:
        return pos in self._counter_held

    def is_counter_available(self, pos: Tuple[int, int]) -> bool:
        return pos not in self._counter_held

    # ---------- Cooker ----------

    def apply_cooker(
        self,
        applier_location: Optional[Tuple[int, int]] = None,
        time_limit: Optional[int] = None,
        specific_pos: Optional[Tuple[int, int]] = None,
    ) -> Tuple[bool, Optional[Tuple[int, int]], CookerStatus]:
        """
        Request exclusive use of a cooker. Resource is held (no owner tracked).
        - applier_location: (x, y); if given, returns closest available cooker
        - time_limit: turns until auto-release (None = no limit)
        - specific_pos: if given, request that cooker only (ignores applier_location)
        Returns (success, pos, status). If no resource available: (False, None, NO_PAN).
        Auto-release also happens when burnt or pan missing (checked in update()).
        """
        if specific_pos is not None:
            if specific_pos not in self._cookers or specific_pos in self._cooker_held:
                candidates = []
            elif applier_location is not None and self._bfs_distance_to_resource(applier_location, specific_pos) is None:
                candidates = []  # not accessible from applier
            else:
                candidates = [specific_pos]
        else:
            pos = self._find_closest_available(
                self._cookers, self._cooker_held, applier_location)
            candidates = [pos] if pos is not None else []
        if not candidates:
            return (False, None, CookerStatus.NO_PAN)
        pos = candidates[0]
        if pos in self._cooker_held:
            return (False, None, self._cooker_status.get(pos, CookerStatus.NO_PAN))
        self._cooker_held.add(pos)
        self._cooker_hold_meta[pos] = {
            "applied_turn": self._current_turn, "time_limit": time_limit}
        return (True, pos, self._cooker_status.get(pos, CookerStatus.NO_PAN))

    def release_cooker(self, pos: Tuple[int, int]) -> bool:
        """Release the cooker. Returns True if it was held."""
        if pos in self._cooker_held:
            self._cooker_held.discard(pos)
            self._cooker_hold_meta.pop(pos, None)
            return True
        return False

    def get_cooker_status(self, pos: Tuple[int, int]) -> CookerStatus:
        return self._cooker_status.get(pos, CookerStatus.NO_PAN)

    def is_cooker_held(self, pos: Tuple[int, int]) -> bool:
        return pos in self._cooker_held

    def is_cooker_available(self, pos: Tuple[int, int]) -> bool:
        return pos not in self._cooker_held

    # ---------- Getters ----------

    def get_all_cookers(self) -> List[Tuple[int, int]]:
        return list(self._cookers)

    def get_all_counters(self) -> List[Tuple[int, int]]:
        return list(self._counters)

    def find_nearest_tile(
        self, controller: RobotController, bot_x: int, bot_y: int, tile_name: str
    ) -> Optional[Tuple[int, int]]:
        """
        Find the nearest tile of the given name reachable from (bot_x, bot_y) by BFS.
        Uses map walkability; returns the tile position when a walkable cell
        adjacent to that tile is first reached.
        """
        m = controller.get_map(controller.get_team())
        if self._is_walkable is None:
            return None
        width, height = m.width, m.height
        targets = [
            (x, y) for x in range(width) for y in range(height)
            if m.tiles[x][y].tile_name == tile_name
        ]
        if not targets:
            return None
        targets_set = set(targets)

        if not (0 <= bot_x < self._width and 0 <= bot_y < self._height):
            return None
        if not self._is_walkable(bot_x, bot_y):
            return None

        for dx, dy in _BFS_NEIGHBORS:
            ax, ay = bot_x + dx, bot_y + dy
            if (ax, ay) in targets_set:
                return (ax, ay)
        if (bot_x, bot_y) in targets_set:
            return (bot_x, bot_y)

        queue: deque[Tuple[int, int]] = deque([(bot_x, bot_y)])
        visited: Set[Tuple[int, int]] = {(bot_x, bot_y)}
        while queue:
            x, y = queue.popleft()
            for dx, dy in _BFS_NEIGHBORS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self._width and 0 <= ny < self._height) or (nx, ny) in visited:
                    continue
                if not self._is_walkable(nx, ny):
                    continue
                visited.add((nx, ny))
                for tx, ty in targets:
                    if max(abs(nx - tx), abs(ny - ty)) <= 1:
                        return (tx, ty)
                queue.append((nx, ny))
        return None

# =============================================================================
# Action Types
# =============================================================================


class ActionType(Enum):
    """Types of single actions a bot can perform."""
    MOVE = auto()                  # Move towards a target
    BUY = auto()                   # Buy from shop
    PLACE = auto()                 # Place item on tile
    PICKUP = auto()                # Pick up item from tile
    CHOP = auto()                  # Chop food on counter
    TAKE_FROM_PAN = auto()         # Take food from pan
    ADD_TO_PLATE = auto()          # Add food to plate
    TRASH = auto()                 # Trash food
    SUBMIT = auto()                # Submit order


# =============================================================================
# Action Class
# =============================================================================

@dataclass
class Action:
    """
    Represents a single atomic action.

    Attributes:
        action_type: The type of action to perform
        food_type: The food type this action relates to (if applicable)
        item_type: Item type (for buying plates/pans)
    """
    action_type: ActionType
    food_type: Optional[FoodType] = None
    item_type: Optional[ShopCosts] = None

    def __repr__(self) -> str:
        return f"Action({self.action_type.name})"


# =============================================================================
# Session Types
# =============================================================================

class SessionType(Enum):
    """
    Types of sessions - atomic units of work that start and end with empty hands.
    """
    # Setup sessions
    BUY_AND_PLACE_PAN = auto()           # Buy pan -> place on cooker
    BUY_AND_PLACE_PLATE = auto()         # Buy plate -> place on counter

    # Food processing sessions
    BUY_AND_PLATE_SIMPLE = auto()        # Buy food -> add to plate (no processing)
    # Buy -> place -> chop
    BUY_CHOP = auto()
    # Buy -> place -> chop -> pickup -> place in cooker
    BUY_CHOP_AND_COOK = auto()
    BUY_AND_COOK = auto()                # Buy -> place in cooker (no chop needed)
    TAKE_AND_PLATE_COOKED = auto()       # Take from pan -> add to plate (when ready)

    # Trash sessions
    CLEAR_PAN = auto()                  # Pick food from pan -> trash
    CLEAR_AND_PLACE_PLATE = auto()      # Clear plate -> place in sink
    TAKE_AND_TRASH = auto()             # Take from counter -> trash

    # Final session
    PICKUP_AND_SUBMIT = auto()           # Pickup plate -> submit


# =============================================================================
# Session Class
# =============================================================================

@dataclass
class Session:
    """
    A Session is an atomic unit of work that starts and ends with the bot
    having empty hands. This makes scheduling easier as sessions don't leave
    the bot in intermediate states.

    Attributes:
        session_type: The type of session
        actions: Queue of actions that make up this session
        food_type: The food type being processed (if applicable)
        completed: Whether this session is finished
        current_action_index: Index of current action being executed
        available_after_turn: Turn after which this session becomes available (for cooking)
        priority: Lower = higher priority (cooking sessions should run first)
    """
    session_type: SessionType
    actions: List[Action] = field(default_factory=list)
    food_type: Optional[FoodType] = None
    completed: bool = False
    current_action_index: int = 0
    available_after_turn: Optional[int] = None  # For TAKE_AND_PLATE_COOKED
    priority: int = 10  # Lower = do first

    def is_available(self, current_turn: int) -> bool:
        """Check if this session is available to execute."""
        if self.completed:
            return False
        if self.available_after_turn is not None:
            return current_turn >= self.available_after_turn
        return True

    def get_current_action(self) -> Optional[Action]:
        """Get the current action to execute."""
        if self.current_action_index < len(self.actions):
            return self.actions[self.current_action_index]
        return None

    def advance_action(self) -> bool:
        """Move to next action. Returns True if session completed."""
        self.current_action_index += 1
        if self.current_action_index >= len(self.actions):
            self.completed = True
            return True
        return False

    def reset(self) -> None:
        """Reset session to beginning."""
        self.current_action_index = 0
        self.completed = False

    @staticmethod
    def create_buy_and_place_pan() -> 'Session':
        """Create session: Buy pan -> place on cooker. Target coords set dynamically at action time."""
        return Session(
            session_type=SessionType.BUY_AND_PLACE_PAN,
            actions=[
                Action(ActionType.BUY, item_type=ShopCosts.PAN),
                Action(ActionType.PLACE),
            ],
            priority=0  # Highest priority - setup first
        )

    @staticmethod
    def create_clear_pan() -> 'Session':
        """Create session: Pick food from pan -> trash. Target coords set dynamically at action time."""
        return Session(
            session_type=SessionType.CLEAR_PAN,
            actions=[
                Action(ActionType.TAKE_FROM_PAN),
                Action(ActionType.TRASH),
            ],
            priority=1  # Second highest priority - clear pan second
        )

    @staticmethod
    def create_clear_and_place_plate() -> 'Session':
        """Create session: Clear plate -> place in sink. Target coords set dynamically at action time."""
        return Session(
            session_type=SessionType.CLEAR_AND_PLACE_PLATE,
            actions=[
                Action(ActionType.PICKUP),
                Action(ActionType.PLACE),
            ],
            priority=1  # Second highest priority - clear plate second
        )

    @staticmethod
    def create_buy_and_place_plate() -> 'Session':
        """Create session: Buy plate -> place on counter. Target coords set dynamically at action time."""
        return Session(
            session_type=SessionType.BUY_AND_PLACE_PLATE,
            actions=[
                Action(ActionType.BUY, item_type=ShopCosts.PLATE),
                Action(ActionType.PLACE),
            ],
            priority=1  # After cooking starts, before adding food
        )

    @staticmethod
    def create_take_and_trash() -> 'Session':
        """Create session: Take from counter -> trash. Target coords set dynamically at action time."""
        return Session(
            session_type=SessionType.TAKE_AND_TRASH,
            actions=[
                Action(ActionType.PICKUP),
                Action(ActionType.TRASH),
            ],
            priority=2  # After cooking starts, before adding food
        )

    @staticmethod
    def create_buy_chop_and_cook(food_type: FoodType) -> 'Session':
        """Create session: Buy -> place -> chop -> pickup -> place in cooker. Target coords set dynamically at action time."""
        return Session(
            session_type=SessionType.BUY_CHOP_AND_COOK,
            actions=[
                Action(ActionType.BUY, food_type=food_type),
                Action(ActionType.PLACE),
                Action(ActionType.CHOP),
                Action(ActionType.PICKUP),
                Action(ActionType.PLACE),
            ],
            food_type=food_type,
            priority=3  # High priority - start cooking early
        )

    @staticmethod
    def create_buy_and_cook(food_type: FoodType) -> 'Session':
        """Create session: Buy -> place in cooker (no chop). Target coords set dynamically at action time."""
        return Session(
            session_type=SessionType.BUY_AND_COOK,
            actions=[
                Action(ActionType.BUY, food_type=food_type),
                Action(ActionType.PLACE),
            ],
            food_type=food_type,
            priority=4  # High priority - start cooking early
        )

    @staticmethod
    def create_take_and_plate_cooked(food_type: FoodType) -> 'Session':
        """Create session: Take from pan -> add to plate (when cooked). Target coords set dynamically at action time."""
        return Session(
            session_type=SessionType.TAKE_AND_PLATE_COOKED,
            actions=[
                Action(ActionType.TAKE_FROM_PAN),
                Action(ActionType.ADD_TO_PLATE),
            ],
            food_type=food_type,
            available_after_turn=None,  # Will be set when cooking starts
            priority=5  # Lower priority - do after other prep
        )

    @staticmethod
    def create_buy_chop(food_type: FoodType) -> 'Session':
        """Create session: Buy -> place -> chop. Target coords set dynamically at action time."""
        return Session(
            session_type=SessionType.BUY_CHOP,
            actions=[
                Action(ActionType.BUY, food_type=food_type),
                Action(ActionType.PLACE),
                Action(ActionType.CHOP),
            ],
            food_type=food_type,
            priority=6
        )

    @staticmethod
    def create_buy_and_plate_simple(food_type: FoodType) -> 'Session':
        """Create session: Buy food -> add to plate (no processing needed). Target coords set dynamically at action time."""
        return Session(
            session_type=SessionType.BUY_AND_PLATE_SIMPLE,
            actions=[
                Action(ActionType.BUY, food_type=food_type),
                Action(ActionType.ADD_TO_PLATE),
            ],
            food_type=food_type,
            priority=7  # Normal priority
        )

    @staticmethod
    def create_pickup_and_submit() -> 'Session':
        """Create session: Pickup plate -> submit order. Target coords set dynamically at action time."""
        return Session(
            session_type=SessionType.PICKUP_AND_SUBMIT,
            actions=[
                Action(ActionType.PICKUP),
                Action(ActionType.SUBMIT),
            ],
            priority=100  # Last
        )


# =============================================================================
# FoodMaterial Class (extends Food concept)
# =============================================================================

@dataclass
class FoodMaterial:
    """
    Extends the Food concept with sessions needed to prepare it.

    Attributes:
        food_type: The FoodType enum for this ingredient
        sessions: List of sessions needed to prepare this food
        is_prepared: Whether all sessions are completed
    """
    food_type: FoodType
    sessions: List[Session] = field(default_factory=list)
    current_session_index: int = 0
    is_prepared: bool = False

    @property
    def food_name(self) -> str:
        return self.food_type.food_name

    @property
    def can_chop(self) -> bool:
        return self.food_type.can_chop

    @property
    def can_cook(self) -> bool:
        return self.food_type.can_cook

    @property
    def buy_cost(self) -> int:
        return self.food_type.buy_cost

    def build_sessions(self) -> None:
        """
        Build the session list based on food type requirements.
        Target locations are set dynamically when executing each action.

        Session patterns:
        - Simple (no chop, no cook): BUY_AND_PLATE_SIMPLE
        - Chop only: BUY_CHOP_AND_PLATE
        - Cook only: BUY_AND_COOK + TAKE_AND_PLATE_COOKED
        - Chop + Cook: BUY_CHOP_AND_COOK + TAKE_AND_PLATE_COOKED
        """
        self.sessions.clear()
        self.current_session_index = 0

        if self.can_chop and self.can_cook:
            # Chop then cook (e.g., MEAT)
            self.sessions.append(
                Session.create_buy_chop_and_cook(self.food_type))
            self.sessions.append(
                Session.create_take_and_plate_cooked(self.food_type))

        elif self.can_cook:
            # Cook only (e.g., EGG)
            self.sessions.append(Session.create_buy_and_cook(self.food_type))
            self.sessions.append(
                Session.create_take_and_plate_cooked(self.food_type))

        elif self.can_chop:
            # Chop only (e.g., ONIONS)
            self.sessions.append(
                Session.create_buy_chop(self.food_type))

        else:
            # Simple - no processing (e.g., NOODLES, SAUCE)
            self.sessions.append(
                Session.create_buy_and_plate_simple(self.food_type))

    def get_current_session(self) -> Optional[Session]:
        """Get the current session to execute."""
        if self.current_session_index < len(self.sessions):
            return self.sessions[self.current_session_index]
        return None

    def advance_session(self) -> bool:
        """Move to next session. Returns True if all sessions completed."""
        self.current_session_index += 1
        if self.current_session_index >= len(self.sessions):
            self.is_prepared = True
            return True
        return False

    def total_estimated_time(self) -> int:
        """Count remaining actions across remaining sessions (rough workload measure for priority)."""
        return sum(
            len(s.actions) - s.current_action_index
            for s in self.sessions[self.current_session_index:]
        )


# =============================================================================
# DIYOrder Class (extends Order concept)
# =============================================================================

@dataclass
class DIYOrder:
    """
    Extended Order class containing FoodMaterial items with action queues.

    Attributes:
        order_id: Unique identifier for the order
        required_foods: List of FoodMaterial objects needed for this order
        created_turn: Turn when the order was created
        expires_turn: Turn when the order expires
        reward: Money earned on completion
        penalty: Money lost on expiration
        priority: Calculated priority for scheduling (lower = higher priority)
        is_completed: Whether the order has been submitted
    """
    order_id: int
    required_foods: List[FoodMaterial] = field(default_factory=list)
    created_turn: int = 0
    expires_turn: int = 0
    reward: int = 0
    penalty: int = 0
    priority: float = 0.0
    start: bool = False
    is_completed: bool = False
    # Counter reserved for this order
    reserved_counter_pos: Optional[Tuple[int, int]] = None
    reserved_pan_pos: Optional[Tuple[int, int]] = None
    # Only this bot works on this order (one robot per order)
    assigned_bot_id: Optional[int] = None

    order_sessions: List[Session] = field(default_factory=list)

    @classmethod
    def from_api_order(cls, order_dict: Dict[str, Any]) -> 'DIYOrder':
        """
        Create a DIYOrder from the API order dictionary.
        Locations are resolved dynamically when executing each action.

        Args:
            order_dict: Order dictionary from controller.get_orders()
        """
        # Map food names to FoodType enum
        food_name_to_type = {
            "EGG": FoodType.EGG,
            "ONIONS": FoodType.ONIONS,
            "MEAT": FoodType.MEAT,
            "NOODLES": FoodType.NOODLES,
            "SAUCE": FoodType.SAUCE,
        }

        required_foods = []
        for food_name in order_dict.get("required", []):
            food_type = food_name_to_type.get(food_name)
            if food_type:
                food_material = FoodMaterial(food_type=food_type)
                food_material.build_sessions()
                required_foods.append(food_material)

        diy_order = cls(
            order_id=order_dict.get("order_id", 0),
            required_foods=required_foods,
            created_turn=order_dict.get("created_turn", 0),
            expires_turn=order_dict.get("expires_turn", 0),
            reward=order_dict.get("reward", 0),
            penalty=order_dict.get("penalty", 0),
        )
        return diy_order

    def calculate_priority(self, current_turn: int = 0) -> None:
        """
        Calculate order priority based on urgency and value.
        Lower priority value = more urgent/important.

        Factors:
        - Time until expiration (less time = higher priority)
        - Reward/penalty ratio
        - Estimated completion time
        """
        time_remaining = self.expires_turn - current_turn
        estimated_time = self.total_estimated_time()

        # Urgency factor: how much buffer time do we have?
        buffer_time = time_remaining - estimated_time

        # Value factor: reward per unit of work
        value_ratio = self.reward / max(1, estimated_time)

        # Priority formula: urgency weighted heavily, value as tiebreaker
        # Negative buffer = very urgent (low/negative priority value)
        self.priority = buffer_time - \
            (value_ratio * 0.1) - (self.penalty * 0.05)

    def total_estimated_time(self) -> int:
        """Calculate total estimated time to complete all foods in order."""
        return sum(food.total_estimated_time() for food in self.required_foods)

    def is_expired(self, current_turn: int) -> bool:
        """Check if order has expired."""
        return current_turn > self.expires_turn

    def is_active(self, current_turn: int) -> bool:
        """Check if order is still active."""
        return (self.created_turn <= current_turn <= self.expires_turn
                and not self.is_completed)

    def all_foods_prepared(self) -> bool:
        """Check if all foods in the order are prepared."""
        return all(food.is_prepared for food in self.required_foods)

    def get_all_sessions(self) -> List[Session]:
        """Get all sessions from all foods, flattened into a single list."""
        sessions = []
        for food in self.required_foods:
            sessions.extend(food.sessions)
        return sessions

    def get_next_session(self, current_turn: int) -> Optional[Tuple[Session, Optional[FoodMaterial]]]:
        """
        Get the next available session based on priority and availability.
        Returns the highest priority session that is available now.
        """
        available_sessions = []
        for food in self.required_foods:
            for session in food.sessions:
                if not session.completed and session.is_available(current_turn):
                    available_sessions.append((session, food))

        for session in self.order_sessions:
            if not session.completed and session.is_available(current_turn):
                available_sessions.append((session, None))

        if not available_sessions:
            return None

        # Sort by priority (lower = higher priority)
        available_sessions.sort(key=lambda s: s[0].priority)
        return available_sessions[0]

    def add_order_session(self, session: Session) -> None:
        """Add a session to the order."""
        self.order_sessions.append(session)
        self.order_sessions.sort(key=lambda s: s.priority)

    def pop_order_session(self) -> Optional[Session]:
        """Pop the next session from the order."""
        if self.order_sessions:
            return self.order_sessions.pop(0)
        return None

    def get_current_food(self) -> Optional[FoodMaterial]:
        """Get the food currently being worked on."""
        for food in self.required_foods:
            if not food.is_prepared:
                return food
        return None


# =============================================================================
# Scheduler Class
# =============================================================================

class Scheduler:
    """
    Scheduler that manages orders and assigns work to bots.

    Call receive_orders() each turn to update the order list.
    Orders are maintained in a priority-sorted list.
    """

    def __init__(self):
        self.orders: List[DIYOrder] = []
        self.completed_orders: List[DIYOrder] = []
        self.known_order_ids: set = set()

        self.resource_manager: Optional[ResourceManager] = None

    def receive_orders(self, controller: RobotController, current_turn: int) -> None:
        """
        Called each turn to receive and process new orders.

        Args:
            controller: The RobotController to get orders from
            current_turn: The current game turn
        """
        api_orders = controller.get_orders(controller.get_team())

        for order_dict in api_orders:
            order_id = order_dict.get("order_id")
            is_active = order_dict.get("is_active", False)

            # Skip already known or inactive orders
            if order_id in self.known_order_ids:
                continue
            if not is_active:
                continue

            # Create new DIYOrder and add to list (locations resolved at action time)
            diy_order = DIYOrder.from_api_order(order_dict)

            # Skip unprofitable orders: if (reward - cost) < penalty, ignore
            total_cost = sum(food.buy_cost for food in diy_order.required_foods)
            profit = diy_order.reward - total_cost
            if profit < diy_order.penalty:
                self.known_order_ids.add(order_id)  # Mark as known to avoid re-processing
                continue

            diy_order.calculate_priority(current_turn)
            self.orders.append(diy_order)
            self.known_order_ids.add(order_id)

        # Update priorities and sort
        self._update_priorities(current_turn)
        self._sort_orders()

        # Remove expired orders
        self._remove_expired_orders(current_turn)

    def _update_priorities(self, current_turn: int) -> None:
        """Recalculate priorities for all orders."""
        for order in self.orders:
            order.calculate_priority(current_turn)

    def _sort_orders(self) -> None:
        """Sort orders by priority (lowest first = most urgent)."""
        self.orders.sort(key=lambda o: o.priority)

    def _remove_expired_orders(self, current_turn: int) -> None:
        """Remove orders that have expired. Release reserved counter for each."""
        active_orders = []
        for order in self.orders:
            if order.is_expired(current_turn):
                if order.reserved_counter_pos and self.resource_manager:
                    self.resource_manager.release_counter(
                        order.reserved_counter_pos)
                order.is_completed = True
                order.assigned_bot_id = None
                self.completed_orders.append(order)
            else:
                active_orders.append(order)
        self.orders = active_orders

    def get_highest_order(self) -> Optional[DIYOrder]:
        """Get the highest priority order."""
        for order in self.orders:
            if not order.is_completed:
                return order
        return None

    def get_order_by_id(self, order_id: int) -> Optional[DIYOrder]:
        """Find an order by its ID."""
        for order in self.orders:
            if order.order_id == order_id:
                return order
        return None

    def start_order(self, order_id: int, start_location: Tuple[int, int] = None,
                    bot_id: Optional[int] = None, current_budget: int = 0) -> bool:
        """Start an order. If bot_id is given, assigns this order to that bot only."""
        order = self.get_order_by_id(order_id)
        if order and not order.start:
            # Skip if total cost exceeds current budget
            total_cost = sum(food.buy_cost for food in order.required_foods)
            if total_cost > current_budget:
                return False
            # Reserve counter
            reserve_flag = True
            if order.reserved_counter_pos is None and self.resource_manager:
                ok, pos, _ = self.resource_manager.apply_counter(
                    applier_location=start_location, time_limit=order.expires_turn - order.created_turn)
                if ok and pos is not None:
                    order.reserved_counter_pos = pos
                else:
                    reserve_flag = False
            else:
                if not self.resource_manager:
                    reserve_flag = False
            # Reserve pan when the required food is using pan
            for food in order.required_foods:
                if food.can_cook:
                    if order.reserved_pan_pos is None and self.resource_manager:
                        ok, pos, _ = self.resource_manager.apply_cooker(
                            applier_location=start_location, time_limit=order.expires_turn - order.created_turn)
                        if ok and pos is not None:
                            order.reserved_pan_pos = pos
                        else:
                            reserve_flag = False
                    elif not self.resource_manager:
                        reserve_flag = False
                    break

            # Release every reserved resource if not reserve_flag
            if not reserve_flag:
                if order.reserved_counter_pos is not None and self.resource_manager:
                    self.resource_manager.release_counter(
                        order.reserved_counter_pos)
                    order.reserved_counter_pos = None
                if order.reserved_pan_pos is not None and self.resource_manager:
                    self.resource_manager.release_cooker(
                        order.reserved_pan_pos)
                    order.reserved_pan_pos = None
                return False

            order.start = True
            if bot_id is not None:
                order.assigned_bot_id = bot_id
            return True
        return False

    def complete_order(self, order_id: int) -> bool:
        """Mark an order as completed and release its reserved counter."""
        order = self.get_order_by_id(order_id)
        if order:
            if order.reserved_counter_pos and self.resource_manager:
                self.resource_manager.release_counter(
                    order.reserved_counter_pos)
            if order.reserved_pan_pos and self.resource_manager:
                self.resource_manager.release_cooker(
                    order.reserved_pan_pos)
            order.is_completed = True
            order.assigned_bot_id = None
            self.orders.remove(order)
            self.completed_orders.append(order)
            return True
        return False

    def get_active_orders(self) -> List[DIYOrder]:
        """Get all active (started, non-completed, non-expired) orders."""
        return [o for o in self.orders if o.start and not o.is_completed]

    def get_active_orders_for_bot(self, bot_id: int) -> List[DIYOrder]:
        """Get active orders this bot may work on: unassigned or assigned to this bot."""
        return [o for o in self.orders if o.start and not o.is_completed
                and (o.assigned_bot_id is None or o.assigned_bot_id == bot_id)]

    def get_highest_pending_order(self) -> Optional[DIYOrder]:
        """Get the highest priority order that has not been started (for claiming)."""
        for order in self.orders:
            if not order.start and not order.is_completed:
                return order
        return None

    def get_pending_orders(self) -> List[DIYOrder]:
        """Get all orders that have not been started."""
        return [o for o in self.orders if not o.start and not o.is_completed]

    def __len__(self) -> int:
        return len(self.orders)

    def __repr__(self) -> str:
        return f"Scheduler(orders={len(self.orders)}, completed={len(self.completed_orders)})"

# =============================================================================
# BotWorker Class - Tracks individual bot's current work state
# =============================================================================


@dataclass
class BotWorker:
    """
    Tracks a bot's current work assignment and session progress.
    """
    bot_id: int
    current_order: Optional[DIYOrder] = None
    current_session: Optional[Session] = None
    in_ensure_plate: bool = False
    in_ensure_pan: bool = False
    in_submit_order: bool = False

    def is_busy(self) -> bool:
        return self.in_ensure_plate or self.in_ensure_pan or self.in_submit_order or self.current_session is not None

    def finish_order(self) -> None:
        """Mark order as complete and reset to idle."""
        if self.current_order:
            self.current_order.is_completed = True
        self.current_order = None
        self.current_session = None
        self.in_ensure_plate = False
        self.in_ensure_pan = False
        self.in_submit_order = False

    def clear_state(self) -> None:
        self.current_order = None
        self.current_session = None
        self.in_ensure_plate = False
        self.in_ensure_pan = False
        self.in_submit_order = False


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy

        # Scheduler (no pre-asserted tile locations; resolve dynamically at action time)
        self.scheduler = Scheduler()
        self.resource_manager: Optional[ResourceManager] = None

        # Workers for each bot
        self.workers: Dict[int, BotWorker] = {}

        # Route calculator for path reservation
        self.route_calc: Optional[RouteCalculator] = None

    def _init_route_calculator(self, controller: RobotController) -> None:
        """Initialize or update the route calculator with current map."""
        m = controller.get_map(controller.get_team())

        if self.route_calc is None:
            self.route_calc = RouteCalculator(
                m.width, m.height, m.is_tile_walkable)
        else:
            self.route_calc.update_map(m.width, m.height, m.is_tile_walkable)

    def _init_resource_manager(self, controller: RobotController) -> None:
        """Initialize the resource manager."""
        if not self.resource_manager:
            self.resource_manager = ResourceManager()
            self.resource_manager.discover_resources(controller)
            self.resource_manager.update(controller)
        else:
            self.resource_manager.update(controller)
        self.scheduler.resource_manager = self.resource_manager

    def move_towards(self, controller: RobotController, bot_id: int,
                     target_x: int, target_y: int) -> bool:
        """
        Move bot towards target using RouteCalculator. Pathfinding only requires
        reaching a cell that touches the target (on or adjacent), so the target
        tile does not need to be walkable. Returns True when adjacent to target
        (so bot can interact).
        """
        bot_state = controller.get_bot_state(bot_id)
        bx, by = bot_state['x'], bot_state['y']

        # Check if already adjacent to target (can touch / interact)
        if max(abs(bx - target_x), abs(by - target_y)) <= 1:
            # Release any existing reservation since we've arrived
            if self.route_calc:
                self.route_calc.release_path(bot_id)
            return True

        if self.route_calc is None:
            self._init_route_calculator(controller)

        # Block cells occupied by other robots so we don't step on them
        team = controller.get_team()
        other_robot_cells: Set[Tuple[int, int]] = set()
        for other_id in controller.get_team_bot_ids(team):
            if other_id == bot_id:
                continue
            state = controller.get_bot_state(other_id)
            if state is not None:
                other_robot_cells.add((state["x"], state["y"]))

        # Check if we already have a reservation for this same target
        if self.route_calc.has_reservation(bot_id):
            if self.route_calc.has_reservation(bot_id, (target_x, target_y)):
                self.route_calc.release_path(bot_id)

        if not self.route_calc.has_reservation(bot_id, (target_x, target_y)):
            success, path = self.route_calc.try_reserve_path(
                bot_id, (bx, by), (target_x, target_y),
                extra_blocked_cells=other_robot_cells,
            )
            if not success:
                return False

        step = self.route_calc.get_next_step(bot_id)
        if step:
            dx, dy = step
            if controller.move(bot_id, dx, dy):
                # After moving, check if we're now adjacent to target (can act this turn)
                new_bx, new_by = bx + dx, by + dy
                if max(abs(new_bx - target_x), abs(new_by - target_y)) <= 1:
                    if self.route_calc:
                        self.route_calc.release_path(bot_id)
                    return True
                return False  # Moved one step but not adjacent yet
        else:
            # No more steps - release and re-plan
            self.route_calc.release_path(bot_id)
        return False

    def _resolve_target_for_action(self, controller: RobotController, bot_id: int,
                                   session: Session, order: DIYOrder) -> Optional[Tuple[int, int]]:
        """Resolve dynamic target coord for the current action in the session."""
        action = session.get_current_action()
        if action is None:
            return None
        bot_info = controller.get_bot_state(bot_id)
        bx, by = bot_info['x'], bot_info['y']
        idx = session.current_action_index
        st = session.session_type
        at = action.action_type

        tile_name: Optional[str] = None
        if at == ActionType.BUY:
            tile_name = 'SHOP'
        elif at == ActionType.PLACE:
            if st == SessionType.BUY_AND_PLACE_PAN:
                tile_name = 'COOKER'
            elif st == SessionType.BUY_AND_PLACE_PLATE:
                tile_name = 'COUNTER'
            elif st == SessionType.CLEAR_AND_PLACE_PLATE:
                tile_name = 'SINK'
            elif st == SessionType.BUY_CHOP:
                tile_name = 'COUNTER'
            elif st == SessionType.BUY_CHOP_AND_COOK:
                tile_name = 'COOKER' if idx == 4 else 'COUNTER'
            elif st == SessionType.BUY_AND_COOK:
                tile_name = 'COOKER'
        elif at == ActionType.CHOP or at == ActionType.PICKUP or at == ActionType.ADD_TO_PLATE:
            tile_name = 'COUNTER'
        elif at == ActionType.TAKE_FROM_PAN:
            tile_name = 'COOKER'
        elif at == ActionType.TRASH:
            tile_name = 'TRASH'
        elif at == ActionType.SUBMIT:
            tile_name = 'SUBMIT'

        if tile_name is None:
            return None

        if tile_name == 'COOKER':
            return order.reserved_pan_pos
        elif tile_name == 'COUNTER':
            return order.reserved_counter_pos

        if self.resource_manager is None:
            return None
        return self.resource_manager.find_nearest_tile(controller, bx, by, tile_name)

    def execute_action(self, controller: RobotController, bot_id: int,
                       action: Action, target: Tuple[int, int]) -> Tuple[bool, str]:
        """
        Execute a single action at the given target location.
        Returns (True, message) if action completed, (False, message) with reason otherwise.
        """
        bot_info = controller.get_bot_state(bot_id)
        tx, ty = target

        if action.action_type == ActionType.BUY:
            if action.item_type is not None:
                buy_name = action.item_type.item_name
            elif action.food_type is not None:
                buy_name = action.food_type.food_name
            else:
                return False, "BUY failed: no item type specified"

            if bot_info.get('holding') and bot_info.get('holding').get('type') == buy_name:
                return True, f"BUY processed: bot is already holding {buy_name}"
            if bot_info.get('holding'):
                return False, f"BUY failed: bot is already holding {buy_name}"
            if self.move_towards(controller, bot_id, tx, ty):
                if action.food_type:
                    cost = action.food_type.buy_cost
                    if controller.get_team_money(controller.get_team()) >= cost:
                        ok = controller.buy(bot_id, action.food_type, tx, ty)
                        return (ok, f"BUY {action.food_type.food_name} at ({tx},{ty}): {'ok' if ok else 'controller refused'}")
                    return False, f"BUY failed: not enough money (need {cost})"
                elif action.item_type:
                    cost = action.item_type.buy_cost
                    if controller.get_team_money(controller.get_team()) >= cost:
                        ok = controller.buy(bot_id, action.item_type, tx, ty)
                        return (ok, f"BUY {getattr(action.item_type, 'item_name', type(action.item_type).__name__)} at ({tx},{ty}): {'ok' if ok else 'controller refused'}")
                    return False, f"BUY failed: not enough money (need {cost})"
            return False, "BUY failed: could not move to target"

        elif action.action_type == ActionType.PLACE:
            if not bot_info.get('holding'):
                return False, "PLACE failed: bot is not holding anything"
            if self.move_towards(controller, bot_id, tx, ty):
                # Hold a plate and the target is a counter
                target_tile = controller.get_tile(
                    team=controller.get_team(), x=tx, y=ty)
                if bot_info.get('holding').get('type') == 'Plate' and target_tile.get('type') == 'COUNTER' and isinstance(target_tile.get('item'), Food):
                    controller.add_food_to_plate(bot_id, tx, ty)
                    return (False, "PLACE processes: food added to plate")
                ok = controller.place(bot_id, tx, ty)
                return (ok, f"PLACE at ({tx},{ty}): {'ok' if ok else 'controller refused'}")
            return False, "PLACE failed: could not move to target"

        elif action.action_type == ActionType.PICKUP:
            if bot_info.get('holding'):
                return False, "PICKUP failed: bot is already holding something"
            if self.move_towards(controller, bot_id, tx, ty):
                ok = controller.pickup(bot_id, tx, ty)
                return (ok, f"PICKUP at ({tx},{ty}): {'ok' if ok else 'controller refused'}")
            return False, "PICKUP failed: could not move to target"

        elif action.action_type == ActionType.CHOP:
            if bot_info.get('holding'):
                return False, "CHOP failed: bot must have empty hands"
            if self.move_towards(controller, bot_id, tx, ty):
                ok = controller.chop(bot_id, tx, ty)
                return (ok, f"CHOP at ({tx},{ty}): {'ok' if ok else 'controller refused'}")
            return False, "CHOP failed: could not move to target"

        elif action.action_type == ActionType.TAKE_FROM_PAN:
            if bot_info.get('holding'):
                return False, "TAKE_FROM_PAN failed: bot is already holding something"
            if self.move_towards(controller, bot_id, tx, ty):
                ok = controller.take_from_pan(bot_id, tx, ty)
                return (ok, f"TAKE_FROM_PAN at ({tx},{ty}): {'ok' if ok else 'controller refused'}")
            return False, "TAKE_FROM_PAN failed: could not move to target"

        elif action.action_type == ActionType.TRASH:
            if self.move_towards(controller, bot_id, tx, ty):
                ok = controller.trash(bot_id, tx, ty)
                return (ok, f"TRASH at ({tx},{ty}): {'ok' if ok else 'controller refused'}")
            return False, "TRASH failed: could not move to target"

        elif action.action_type == ActionType.ADD_TO_PLATE:
            tile = controller.get_tile(controller.get_team(), tx, ty)
            if tile is None:
                return False, "ADD_TO_PLATE failed: no tile at target"
            if isinstance(tile.item, Food):
                if self.move_towards(controller, bot_id, tx, ty):
                    if (bot_info.get('holding') or {}).get('type') == 'Plate':
                        ok = controller.add_food_to_plate(bot_id, tx, ty)
                        return (ok, f"ADD_TO_PLATE at ({tx},{ty}): {'ok' if ok else 'controller refused'}")
                    return False, "ADD_TO_PLATE failed: bot must be holding a plate to add food from counter"
                return False, "ADD_TO_PLATE failed: could not move to target"
            elif isinstance(tile.item, Plate):
                if bot_info.get('holding') and (bot_info.get('holding') or {}).get('type') == 'food':
                    return False, "ADD_TO_PLATE failed: bot must have empty hands to add food to plate on counter"
                if self.move_towards(controller, bot_id, tx, ty):
                    ok = controller.add_food_to_plate(bot_id, tx, ty)
                    return (ok, f"ADD_TO_PLATE at ({tx},{ty}): {'ok' if ok else 'controller refused'}")
                return False, "ADD_TO_PLATE failed: could not move to target"
            return False, "ADD_TO_PLATE failed: target tile has no Food or Plate"

        elif action.action_type == ActionType.SUBMIT:
            if self.move_towards(controller, bot_id, tx, ty):
                ok = controller.submit(bot_id, tx, ty)
                return (ok, f"SUBMIT at ({tx},{ty}): {'ok' if ok else 'controller refused'}")
            return False, "SUBMIT failed: could not move to target"

        return False, f"execute_action: unknown action type {getattr(action.action_type, 'name', action.action_type)}"

    def execute_session(self, controller: RobotController, bot_id: int,
                        session: Session, target: Optional[Tuple[int, int]] = None) -> Tuple[bool, str]:
        """
        Execute the current action in a session. Target is resolved dynamically if not provided.
        Returns (True, message) if the action completed, (False, message) with reason otherwise.
        """
        action = session.get_current_action()
        if action is None:
            return False, "execute_session: no current action in session"
        if target is None:
            return False, "execute_session: target not provided (must be resolved with order context)"
        ok, msg = self.execute_action(controller, bot_id, action, target)
        if ok:
            session.advance_action()
            return True, msg
        return False, msg

    def ensure_pan_on_cooker(self, controller: RobotController, bot_id: int,
                             worker: BotWorker, order: DIYOrder) -> bool:
        """Ensure pan is on cooker. Returns True if ready. Uses dynamic cooker/shop location."""
        bot_info = controller.get_bot_state(bot_id)
        bx, by = bot_info['x'], bot_info['y']
        cooker = order.reserved_pan_pos
        if cooker is None:
            return False
        kx, ky = cooker
        tile = controller.get_tile(controller.get_team(), kx, ky)

        holding = bot_info.get('holding')

        if holding:
            if holding.get('type') == 'Food':
                trash_tile = self.resource_manager.find_nearest_tile(
                    controller, bx, by, 'TRASH')
                if trash_tile is not None:
                    tx, ty = trash_tile
                    if self.move_towards(controller, bot_id, tx, ty):
                        if controller.trash(bot_id, tx, ty):
                            return True
                return False
            if holding.get('type') == 'Pan':
                if self.move_towards(controller, bot_id, kx, ky):
                    if controller.place(bot_id, kx, ky):
                        worker.in_ensure_pan = False
                        return True
                return False

        if tile:
            if tile.item is None:
                shop = self.resource_manager.find_nearest_tile(
                    controller, bx, by, 'SHOP') if self.resource_manager else None
                if shop is None:
                    return False
                sx, sy = shop
                if self.move_towards(controller, bot_id, sx, sy):
                    if controller.get_team_money(controller.get_team()) >= ShopCosts.PAN.buy_cost:
                        controller.buy(bot_id, ShopCosts.PAN, sx, sy)
                return False
            if isinstance(tile.item, Pan):
                if tile.item.food:
                    # Take food from the pan (use take_from_pan, not pickup)
                    if self.move_towards(controller, bot_id, kx, ky):
                        if controller.take_from_pan(bot_id, kx, ky):
                            return False
                    return False
                else:
                    worker.in_ensure_pan = False
                    return True
            else:
                if self.move_towards(controller, bot_id, kx, ky):
                    if controller.pickup(bot_id, kx, ky):
                        return False
        return False

    def ensure_plate_on_counter(self, controller: RobotController, bot_id: int,
                                worker: BotWorker, order: DIYOrder) -> bool:
        """Ensure plate is on counter. Returns True if ready. Uses dynamic counter/shop location."""
        bot_info = controller.get_bot_state(bot_id)
        bx, by = bot_info['x'], bot_info['y']
        counter = order.reserved_counter_pos
        if counter is None:
            return False
        cx, cy = counter
        tile = controller.get_tile(controller.get_team(), cx, cy)

        if tile and isinstance(tile.item, Plate) and not tile.item.dirty:
            worker.in_ensure_plate = False
            return True

        holding = bot_info.get('holding')
        if holding and holding.get('type') == 'Plate':
            if holding.get('dirty'):
                worker.in_ensure_plate = False
                return False
            if self.move_towards(controller, bot_id, cx, cy):
                # Food on the counter
                target_tile = controller.get_tile(
                    team=controller.get_team(), x=cx, y=cy)
                if target_tile is None:
                    return False, "PLACE failed: target tile not found"

                if target_tile.item is not None:
                    if isinstance(target_tile.item, Food):
                        controller.add_food_to_plate(bot_id, cx, cy)
                        return False, "PLACE processes: food added to plate"
                    else:
                        return False, "PLACE failed: target tile is not food"

                if controller.place(bot_id, cx, cy):
                    worker.in_ensure_plate = False
                    return True
        else:
            shop = self.resource_manager.find_nearest_tile(
                controller, bx, by, 'SHOP') if self.resource_manager else None
            if shop is None:
                return False
            sx, sy = shop
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(controller.get_team()) >= ShopCosts.PLATE.buy_cost:
                    controller.buy(bot_id, ShopCosts.PLATE, sx, sy)
        return False

    def submit_order(self, controller: RobotController, bot_id: int, worker: BotWorker, order: DIYOrder) -> bool:
        """Submit the order. Returns True if submit succeeded."""
        bot_info = controller.get_bot_state(bot_id)
        bx, by = bot_info['x'], bot_info['y']

        # Holding a empty plate, so we need to pick up the food on the counter
        if bot_info.get('holding') and bot_info.get('holding').get('type') == 'Plate' and not bot_info.get('holding').get('food'):
            counter = order.reserved_counter_pos
            if counter is not None:
                cx, cy = counter
                if self.move_towards(controller, bot_id, cx, cy):
                    if controller.add_food_to_plate(bot_id, cx, cy):
                        return False
            return False
        
        if bot_info.get('holding') and bot_info.get('holding').get('type') == 'Plate' and bot_info.get('holding').get('food'):
            submit_tile = self.resource_manager.find_nearest_tile(
                controller, bx, by, 'SUBMIT')
            if submit_tile is not None:
                sx, sy = submit_tile
                if self.move_towards(controller, bot_id, sx, sy):
                    if controller.submit(bot_id, sx, sy):
                        worker.in_submit_order = False
                        return True
                return False
            worker.in_submit_order = False
        
        # The counter is not a plate, so we need to buy a plate
        counter = order.reserved_counter_pos
        # check counter is a plate
        if counter is not None:
            cx, cy = counter
            plate_tile = controller.get_tile(
                controller.get_team(), cx, cy)
            if plate_tile is None:
                return False
            if not isinstance(plate_tile.item, Plate):
                # buy a plate
                shop = self.resource_manager.find_nearest_tile(
                    controller, bx, by, 'SHOP') if self.resource_manager else None
                if shop is None:
                    return False
                sx, sy = shop
                if self.move_towards(controller, bot_id, sx, sy):
                    if controller.get_team_money(controller.get_team()) >= ShopCosts.PLATE.buy_cost:
                        controller.buy(bot_id, ShopCosts.PLATE, sx, sy)
                return False
            else:
                if self.move_towards(controller, bot_id, cx, cy):
                    if controller.pickup(bot_id, cx, cy):
                        return False
                return False

        return False
    
    def run_bot(self, controller: RobotController, bot_id: int) -> None:
        """Run a single bot's turn using session-based scheduling."""
        turn = controller.get_turn()

        # Get or create worker
        if bot_id not in self.workers:
            self.workers[bot_id] = BotWorker(bot_id=bot_id)
        worker = self.workers[bot_id]

        if worker.current_session is not None and worker.current_session.completed:
            worker.current_session = None
        # If the order assigned to this bot is completed, release the counter and pan
        if worker.current_order is not None:
            if worker.current_order.is_completed or worker.current_order.is_expired(turn):
                self.scheduler.complete_order(worker.current_order.order_id)
                worker.clear_state()

        # If the hand is holding something, trash it
        if not worker.is_busy():
            robot_info = controller.get_bot_state(bot_id)
            bx, by = robot_info.get('x'), robot_info.get('y')
            if robot_info.get('holding'):
                if robot_info.get('holding').get('type') == 'Food':
                    trash_tile = self.resource_manager.find_nearest_tile(
                        controller, bx, by, 'TRASH')
                    if trash_tile is not None:
                        tx, ty = trash_tile
                        if self.move_towards(controller, bot_id, tx, ty):
                            if controller.trash(bot_id, tx, ty):
                                return
                        return
                elif robot_info.get('holding').get('type') == 'Pan' or robot_info.get('holding').get('type') == 'Plate':
                    box_tile = self.resource_manager.find_nearest_tile(
                        controller, bx, by, 'BOX')
                    if box_tile is not None:
                        bx, by = box_tile
                        if self.move_towards(controller, bot_id, bx, by):
                            if controller.place(bot_id, bx, by):
                                return
                        return
                return
        
        if worker.current_order is None:
            order = self.scheduler.get_highest_pending_order()
            if order is not None:
                current_budget = controller.get_team_money(controller.get_team())
                if self.scheduler.start_order(order.order_id, (bx, by), bot_id, current_budget):
                    worker.clear_state()
                    worker.current_order = order
                    foods_str = ", ".join(
                        f.food_name for f in order.required_foods)
                    print(
                        f"[Turn {turn}] Scheduler: Bot {bot_id} started order {order.order_id} ({foods_str})")
                else:
                    return
            else:
                return
        else:
            order = worker.current_order
        
        if order is None:
            worker.clear_state()
            return
        
        # If the bot is ensuring the plate is on the counter, ensure it is ready
        if worker.in_ensure_plate:
            if self.ensure_plate_on_counter(controller, bot_id, worker, worker.current_order):
                worker.in_ensure_plate = False
            return
        
        # If the bot is ensuring the pan is ready, ensure it is ready
        if worker.in_ensure_pan:
            if self.ensure_pan_on_cooker(controller, bot_id, worker, worker.current_order):
                worker.in_ensure_pan = False
            return
        
        # If the bot is submitting the order, submit it
        if worker.in_submit_order:
            if self.submit_order(controller, bot_id, worker, worker.current_order):
                worker.in_submit_order = False
                self.scheduler.complete_order(worker.current_order.order_id)
                print(f"[Turn {turn}] Bot {bot_id}: Order {worker.current_order.order_id} submitted")
                worker.clear_state()
            return
        
        result = order.get_next_session(turn)
        
        session, food = result if result is not None else (None, None)

        if session:
            # Claim this order if not yet assigned (one robot per order)
            if order.assigned_bot_id is None:
                order.assigned_bot_id = bot_id
            # Check if this session needs plate - ensure plate is ready
            needs_plate = session.session_type in [
                SessionType.BUY_AND_PLATE_SIMPLE,
                SessionType.TAKE_AND_PLATE_COOKED,
            ]
            if needs_plate and order.reserved_counter_pos is not None:
                # Check the order reserved counter has a plate
                cx, cy = order.reserved_counter_pos
                plate_tile = controller.get_tile(
                    controller.get_team(), cx, cy)
                if plate_tile is None:
                    print(f"[Turn {turn}] Bot {bot_id}: Plate not found at {cx}, {cy}, ending order {order.order_id}")
                    self.scheduler.complete_order(order.order_id)
                    worker.clear_state()
                    return
                if not isinstance(plate_tile.item, Plate) or plate_tile.item.dirty:
                    print(
                        f"[Turn {turn}] Bot {bot_id}: Plate not ready at {cx}, {cy}")
                    worker.in_ensure_plate = True
                    self.ensure_plate_on_counter(
                        controller, bot_id, worker, order)

            # Check if this session need pan - ensure pan is ready
            needs_pan = session.session_type in [
                SessionType.BUY_CHOP_AND_COOK, SessionType.BUY_AND_COOK]
            if needs_pan and order.reserved_pan_pos is not None:
                cx, cy = order.reserved_pan_pos
                cooker_tile = controller.get_tile(
                    controller.get_team(), cx, cy)
                if cooker_tile is None:
                    print(f"[Turn {turn}] Bot {bot_id}: Pan not found at {cx}, {cy}, ending order {order.order_id}")
                    self.scheduler.complete_order(order.order_id)
                    worker.clear_state()
                    return
                has_empty_pan = (
                    getattr(cooker_tile, "item", None) is not None
                    and isinstance(cooker_tile.item, Pan)
                    and cooker_tile.item.food is None
                )
                if not has_empty_pan:
                    print(
                        f"[Turn {turn}] Bot {bot_id}: Pan not ready at {cx}, {cy}")
                    worker.in_ensure_pan = True
                    self.ensure_pan_on_cooker(
                        controller, bot_id, worker, order)

            worker.current_session = session

            food_name = food.food_name if food else None
            action = session.get_current_action()
            target = self._resolve_target_for_action(
                controller, bot_id, session, order)
            action_str = f"{action.action_type.name} @ {target}" if action else "None"
            print(f"[Turn {turn}] Bot {bot_id}: Session={session.session_type.name}, "
                    f"Food={food_name}, Action={action_str}")

            # Execute current session with dynamic target
            action_ok, action_msg = self.execute_session(
                controller, bot_id, session, target)
            print(f"[Turn {turn}] Bot {bot_id}: {action_msg}")

            if session.completed:
                worker.current_session = None
                print(f"[Turn {turn}] Bot {bot_id}: Session COMPLETED!")

                # Advance the food's session tracking
                if food:
                    food.advance_session()
                    if food.is_prepared:
                        print(
                            f"[Turn {turn}] Bot {bot_id}: {food_name} is fully prepared!")

                # If this was a cooking session, set the cook ready time
                if session.session_type in [SessionType.BUY_CHOP_AND_COOK, SessionType.BUY_AND_COOK]:
                    # Find the TAKE_AND_PLATE_COOKED session for this food
                    if food:
                        for s in food.sessions:
                            if s.session_type == SessionType.TAKE_AND_PLATE_COOKED:
                                s.available_after_turn = turn + GameConstants.COOK_PROGRESS
                                print(
                                    f"[Turn {turn}] Bot {bot_id}: {food_name} will be ready at turn {s.available_after_turn}")
                                break
            return
        else:
            # No available sessions - check if all foods are prepared
            if order.all_foods_prepared():
                # Submit the order
                worker.in_submit_order = True
                print(
                    f"[Turn {turn}] Bot {bot_id}: All foods ready, queuing submit...")
                if self.submit_order(controller, bot_id, worker, order):
                    worker.in_submit_order = False
                    self.scheduler.complete_order(worker.current_order.order_id)
                    print(f"[Turn {turn}] Bot {bot_id}: Order {worker.current_order.order_id} submitted")
                    worker.clear_state()

            else:
                # Nothing to do, just ensure the pan
                cx, cy = order.reserved_counter_pos
                plate_tile = controller.get_tile(
                    controller.get_team(), cx, cy)
                if plate_tile is None:
                    print(f"[Turn {turn}] Bot {bot_id}: Plate not found at {cx}, {cy}, ending order {order.order_id}")
                    self.scheduler.complete_order(order.order_id)
                    return
                if not isinstance(plate_tile.item, Plate) or plate_tile.item.dirty:
                    print(
                        f"[Turn {turn}] Bot {bot_id}: Plate not ready at {cx}, {cy}")
                    worker.in_ensure_plate = True
                    self.ensure_plate_on_counter(
                        controller, bot_id, worker, order)
                    return
                # Waiting for something (cooking)
                print(
                    f"[Turn {turn}] Bot {bot_id}: Waiting (cooking in progress)...")
            return  # One decision per bot per turn

    def play_turn(self, controller: RobotController):
        """Main entry point - called each turn."""
        my_bots = controller.get_team_bot_ids(controller.get_team())
        if not my_bots:
            return

        # Initialize/update route calculator for this turn
        self._init_route_calculator(controller)

        # Update resource manager each round (status, auto-release cookers, etc.)
        self._init_resource_manager(controller)

        # Update scheduler with new orders
        self.scheduler.receive_orders(controller, controller.get_turn())

        # Run each bot
        for bot_id in my_bots:
            self.run_bot(controller, bot_id)

        # Advance route reservations for next turn
        if self.route_calc:
            self.route_calc.advance_round()
