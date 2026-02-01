from collections import deque
from typing import Tuple, Optional, List, Dict, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from game_constants import FoodType, ShopCosts, GameConstants
from robot_controller import RobotController
from item import Pan, Plate

class RouteCalculator:
    """
    Reads the map and maintains path reservations with a per-round timer.
    - If a robot reserves a path, those cells are marked until the robot advances (one step released per round).
    - Another robot that wants to walk searches against this record; if the path is not possible it should give up.
    - If possible, both robots can move (their paths don't conflict).
    """

    # Neighbors for Chebyshev distance 1 (8-direction)
    _NEIGHBORS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

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
        BFS from start to end. A cell is passable if walkable and not in blocked_cells.
        Returns full path [start, ..., end] or None if no path.
        """
        if blocked_cells is None:
            blocked_cells = set()
        sx, sy = start
        ex, ey = end
        if not self._in_bounds(sx, sy) or not self._in_bounds(ex, ey):
            return None
        if not self._is_walkable(sx, sy) or not self._is_walkable(ex, ey):
            return None
        if (sx, sy) == (ex, ey):
            return [start]

        queue: deque[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = deque([(start, [start])])
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
                if (nx, ny) == (ex, ey):
                    return new_path
                queue.append(((nx, ny), new_path))
        return None

    def can_reach(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        exclude_bot_id: Optional[int] = None,
    ) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
        """
        Check if a path exists from start to end given current reservations.
        exclude_bot_id: when computing for one robot, ignore its own reservation.
        Returns (True, path) or (False, None).
        """
        blocked = self.get_reserved_cells(exclude_bot_id=exclude_bot_id)
        path = self.find_path(start, end, blocked_cells=blocked)
        return (path is not None, path)

    def try_reserve_path(
        self,
        bot_id: int,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
        """
        Find a path from start to end considering other robots' reservations.
        If a path exists, reserve it for bot_id and return (True, path).
        If not accessible, return (False, None) — caller should give up.
        """
        ok, path = self.can_reach(start, end, exclude_bot_id=bot_id)
        if not ok or path is None:
            return (False, None)
        self._reservations[bot_id] = {"path": path, "step": 0}
        return (True, path)

    def reserve_path(self, bot_id: int, path: List[Tuple[int, int]]) -> bool:
        """
        Manually reserve a path for bot_id (e.g. one you computed yourself).
        path must be a valid sequence of adjacent cells. Returns True if stored.
        """
        if not path:
            return False
        self._reservations[bot_id] = {"path": list(path), "step": 0}
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

    def has_reservation(self, bot_id: int) -> bool:
        return bot_id in self._reservations

    def is_cell_reserved(self, x: int, y: int, exclude_bot_id: Optional[int] = None) -> bool:
        return (x, y) in self.get_reserved_cells(exclude_bot_id=exclude_bot_id)


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
        target_coord: Target coordinate (x, y) for the action
        food_type: The food type this action relates to (if applicable)
        item_type: Item type (for buying plates/pans)
    """
    action_type: ActionType
    target_coord: Optional[Tuple[int, int]] = None
    food_type: Optional[FoodType] = None
    item_type: Optional[ShopCosts] = None

    def __repr__(self) -> str:
        return f"Action({self.action_type.name}, target={self.target_coord})"


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
    BUY_CHOP_AND_PLATE = auto()          # Buy -> place -> chop -> pickup -> add to plate
    BUY_CHOP_AND_COOK = auto()           # Buy -> place -> chop -> pickup -> place in cooker
    BUY_AND_COOK = auto()                # Buy -> place in cooker (no chop needed)
    TAKE_AND_PLATE_COOKED = auto()       # Take from pan -> add to plate (when ready)

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
        start_coord: Starting coordinate for estimating time
        end_coord: Ending coordinate for estimating time
        completed: Whether this session is finished
        current_action_index: Index of current action being executed
        available_after_turn: Turn after which this session becomes available (for cooking)
        priority: Lower = higher priority (cooking sessions should run first)
    """
    session_type: SessionType
    actions: List[Action] = field(default_factory=list)
    food_type: Optional[FoodType] = None
    start_coord: Optional[Tuple[int, int]] = None
    end_coord: Optional[Tuple[int, int]] = None
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

    def estimate_time(self) -> int:
        """Estimate total time for this session."""
        time = len(self.actions)  # Base: 1 turn per action
        # Add travel time estimate
        if self.start_coord and self.end_coord:
            time += max(abs(self.end_coord[0] - self.start_coord[0]),
                       abs(self.end_coord[1] - self.start_coord[1]))
        return time

    @staticmethod
    def create_buy_and_place_pan(shop_loc: Tuple[int, int],
                                  cooker_loc: Tuple[int, int]) -> 'Session':
        """Create session: Buy pan -> place on cooker."""
        return Session(
            session_type=SessionType.BUY_AND_PLACE_PAN,
            actions=[
                Action(ActionType.BUY, target_coord=shop_loc, item_type=ShopCosts.PAN),
                Action(ActionType.PLACE, target_coord=cooker_loc),
            ],
            start_coord=shop_loc,
            end_coord=cooker_loc,
            priority=0  # Highest priority - setup first
        )

    @staticmethod
    def create_buy_and_place_plate(shop_loc: Tuple[int, int],
                                    counter_loc: Tuple[int, int]) -> 'Session':
        """Create session: Buy plate -> place on counter."""
        return Session(
            session_type=SessionType.BUY_AND_PLACE_PLATE,
            actions=[
                Action(ActionType.BUY, target_coord=shop_loc, item_type=ShopCosts.PLATE),
                Action(ActionType.PLACE, target_coord=counter_loc),
            ],
            start_coord=shop_loc,
            end_coord=counter_loc,
            priority=5  # After cooking starts, before adding food
        )

    @staticmethod
    def create_buy_and_plate_simple(shop_loc: Tuple[int, int],
                                     plate_loc: Tuple[int, int],
                                     food_type: FoodType) -> 'Session':
        """Create session: Buy food -> add to plate (no processing needed)."""
        return Session(
            session_type=SessionType.BUY_AND_PLATE_SIMPLE,
            actions=[
                Action(ActionType.BUY, target_coord=shop_loc, food_type=food_type),
                Action(ActionType.ADD_TO_PLATE, target_coord=plate_loc),
            ],
            food_type=food_type,
            start_coord=shop_loc,
            end_coord=plate_loc,
            priority=10  # Normal priority
        )

    @staticmethod
    def create_buy_chop_and_plate(shop_loc: Tuple[int, int],
                                   counter_loc: Tuple[int, int],
                                   plate_loc: Tuple[int, int],
                                   food_type: FoodType) -> 'Session':
        """Create session: Buy -> place -> chop -> pickup -> add to plate."""
        return Session(
            session_type=SessionType.BUY_CHOP_AND_PLATE,
            actions=[
                Action(ActionType.BUY, target_coord=shop_loc, food_type=food_type),
                Action(ActionType.PLACE, target_coord=counter_loc),
                Action(ActionType.CHOP, target_coord=counter_loc),
                Action(ActionType.PICKUP, target_coord=counter_loc),
                Action(ActionType.ADD_TO_PLATE, target_coord=plate_loc),
            ],
            food_type=food_type,
            start_coord=shop_loc,
            end_coord=plate_loc,
            priority=10
        )

    @staticmethod
    def create_buy_chop_and_cook(shop_loc: Tuple[int, int],
                                  counter_loc: Tuple[int, int],
                                  cooker_loc: Tuple[int, int],
                                  food_type: FoodType) -> 'Session':
        """Create session: Buy -> place -> chop -> pickup -> place in cooker."""
        return Session(
            session_type=SessionType.BUY_CHOP_AND_COOK,
            actions=[
                Action(ActionType.BUY, target_coord=shop_loc, food_type=food_type),
                Action(ActionType.PLACE, target_coord=counter_loc),
                Action(ActionType.CHOP, target_coord=counter_loc),
                Action(ActionType.PICKUP, target_coord=counter_loc),
                Action(ActionType.PLACE, target_coord=cooker_loc),
            ],
            food_type=food_type,
            start_coord=shop_loc,
            end_coord=cooker_loc,
            priority=1  # High priority - start cooking early
        )

    @staticmethod
    def create_buy_and_cook(shop_loc: Tuple[int, int],
                             cooker_loc: Tuple[int, int],
                             food_type: FoodType) -> 'Session':
        """Create session: Buy -> place in cooker (no chop)."""
        return Session(
            session_type=SessionType.BUY_AND_COOK,
            actions=[
                Action(ActionType.BUY, target_coord=shop_loc, food_type=food_type),
                Action(ActionType.PLACE, target_coord=cooker_loc),
            ],
            food_type=food_type,
            start_coord=shop_loc,
            end_coord=cooker_loc,
            priority=1  # High priority - start cooking early
        )

    @staticmethod
    def create_take_and_plate_cooked(cooker_loc: Tuple[int, int],
                                      plate_loc: Tuple[int, int],
                                      food_type: FoodType) -> 'Session':
        """Create session: Take from pan -> add to plate (when cooked)."""
        return Session(
            session_type=SessionType.TAKE_AND_PLATE_COOKED,
            actions=[
                Action(ActionType.TAKE_FROM_PAN, target_coord=cooker_loc),
                Action(ActionType.ADD_TO_PLATE, target_coord=plate_loc),
            ],
            food_type=food_type,
            start_coord=cooker_loc,
            end_coord=plate_loc,
            available_after_turn=None,  # Will be set when cooking starts
            priority=20  # Lower priority - do after other prep
        )

    @staticmethod
    def create_pickup_and_submit(plate_loc: Tuple[int, int],
                                  submit_loc: Tuple[int, int]) -> 'Session':
        """Create session: Pickup plate -> submit order."""
        return Session(
            session_type=SessionType.PICKUP_AND_SUBMIT,
            actions=[
                Action(ActionType.PICKUP, target_coord=plate_loc),
                Action(ActionType.SUBMIT, target_coord=submit_loc),
            ],
            start_coord=plate_loc,
            end_coord=submit_loc,
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

    def build_sessions(self, shop_loc: Tuple[int, int],
                       counter_loc: Tuple[int, int],
                       cooker_loc: Tuple[int, int],
                       plate_loc: Tuple[int, int]) -> None:
        """
        Build the session list based on food type requirements.

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
            self.sessions.append(Session.create_buy_chop_and_cook(
                shop_loc, counter_loc, cooker_loc, self.food_type))
            self.sessions.append(Session.create_take_and_plate_cooked(
                cooker_loc, plate_loc, self.food_type))

        elif self.can_cook:
            # Cook only (e.g., EGG)
            self.sessions.append(Session.create_buy_and_cook(
                shop_loc, cooker_loc, self.food_type))
            self.sessions.append(Session.create_take_and_plate_cooked(
                cooker_loc, plate_loc, self.food_type))

        elif self.can_chop:
            # Chop only (e.g., ONIONS)
            self.sessions.append(Session.create_buy_chop_and_plate(
                shop_loc, counter_loc, plate_loc, self.food_type))

        else:
            # Simple - no processing (e.g., NOODLES, SAUCE)
            self.sessions.append(Session.create_buy_and_plate_simple(
                shop_loc, plate_loc, self.food_type))

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
        """Calculate total estimated time for all remaining sessions."""
        return sum(s.estimate_time() for s in self.sessions[self.current_session_index:])


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
        assigned_bot_id: Bot assigned to complete this order
        is_completed: Whether the order has been submitted
    """
    order_id: int
    required_foods: List[FoodMaterial] = field(default_factory=list)
    created_turn: int = 0
    expires_turn: int = 0
    reward: int = 0
    penalty: int = 0
    priority: float = 0.0
    assigned_bot_id: Optional[int] = None
    is_completed: bool = False

    @classmethod
    def from_api_order(cls, order_dict: Dict[str, Any],
                       shop_loc: Tuple[int, int],
                       counter_loc: Tuple[int, int],
                       cooker_loc: Tuple[int, int],
                       plate_loc: Tuple[int, int]) -> 'DIYOrder':
        """
        Create a DIYOrder from the API order dictionary.

        Args:
            order_dict: Order dictionary from controller.get_orders()
            shop_loc: Location of shop
            counter_loc: Location of counter
            cooker_loc: Location of cooker
            plate_loc: Location for plate assembly
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
                food_material.build_sessions(shop_loc, counter_loc, cooker_loc, plate_loc)
                required_foods.append(food_material)

        diy_order = cls(
            order_id=order_dict.get("order_id", 0),
            required_foods=required_foods,
            created_turn=order_dict.get("created_turn", 0),
            expires_turn=order_dict.get("expires_turn", 0),
            reward=order_dict.get("reward", 0),
            penalty=order_dict.get("penalty", 0),
        )
        diy_order.calculate_priority()
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
        self.priority = buffer_time - (value_ratio * 0.1) - (self.penalty * 0.05)

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

    def get_next_session(self, current_turn: int) -> Optional[Session]:
        """
        Get the next available session based on priority and availability.
        Returns the highest priority session that is available now.
        """
        available_sessions = []
        for food in self.required_foods:
            for session in food.sessions:
                if not session.completed and session.is_available(current_turn):
                    available_sessions.append(session)

        if not available_sessions:
            return None

        # Sort by priority (lower = higher priority)
        available_sessions.sort(key=lambda s: s.priority)
        return available_sessions[0]

    def get_food_for_session(self, session: Session) -> Optional[FoodMaterial]:
        """Get the FoodMaterial that owns this session."""
        for food in self.required_foods:
            if session in food.sessions:
                return food
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

        # Tile locations (to be set by BotPlayer)
        self.shop_loc: Optional[Tuple[int, int]] = None
        self.counter_loc: Optional[Tuple[int, int]] = None
        self.cooker_loc: Optional[Tuple[int, int]] = None
        self.submit_loc: Optional[Tuple[int, int]] = None
        self.plate_loc: Optional[Tuple[int, int]] = None  # Where plate is assembled

    def set_locations(self, shop: Tuple[int, int], counter: Tuple[int, int],
                      cooker: Tuple[int, int], submit: Tuple[int, int]) -> None:
        """Set the tile locations for action planning."""
        self.shop_loc = shop
        self.counter_loc = counter
        self.cooker_loc = cooker
        self.submit_loc = submit
        self.plate_loc = counter  # Default: assemble plate on counter

    def receive_orders(self, controller: RobotController, current_turn: int) -> None:
        """
        Called each turn to receive and process new orders.

        Args:
            controller: The RobotController to get orders from
            current_turn: The current game turn
        """
        if not all([self.shop_loc, self.counter_loc, self.cooker_loc, self.submit_loc]):
            return  # Locations not set yet

        api_orders = controller.get_orders(controller.get_team())

        for order_dict in api_orders:
            order_id = order_dict.get("order_id")
            is_active = order_dict.get("is_active", False)

            # Skip already known or inactive orders
            if order_id in self.known_order_ids:
                continue
            if not is_active:
                continue

            # Create new DIYOrder and add to list
            diy_order = DIYOrder.from_api_order(
                order_dict,
                self.shop_loc,
                self.counter_loc,
                self.cooker_loc,
                self.plate_loc
            )
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
        """Remove orders that have expired."""
        active_orders = []
        for order in self.orders:
            if order.is_expired(current_turn):
                # Move to completed (as failed)
                order.is_completed = True
                self.completed_orders.append(order)
            else:
                active_orders.append(order)
        self.orders = active_orders

    def get_next_order(self) -> Optional[DIYOrder]:
        """Get the highest priority unassigned order."""
        for order in self.orders:
            if order.assigned_bot_id is None and not order.is_completed:
                return order
        return None

    def get_order_by_id(self, order_id: int) -> Optional[DIYOrder]:
        """Find an order by its ID."""
        for order in self.orders:
            if order.order_id == order_id:
                return order
        return None

    def assign_order(self, order_id: int, bot_id: int) -> bool:
        """Assign an order to a specific bot."""
        order = self.get_order_by_id(order_id)
        if order and order.assigned_bot_id is None:
            order.assigned_bot_id = bot_id
            return True
        return False

    def complete_order(self, order_id: int) -> bool:
        """Mark an order as completed."""
        order = self.get_order_by_id(order_id)
        if order:
            order.is_completed = True
            self.orders.remove(order)
            self.completed_orders.append(order)
            return True
        return False

    def get_orders_for_bot(self, bot_id: int) -> List[DIYOrder]:
        """Get all orders assigned to a specific bot."""
        return [o for o in self.orders if o.assigned_bot_id == bot_id]

    def get_active_orders(self) -> List[DIYOrder]:
        """Get all active (non-completed, non-expired) orders."""
        return [o for o in self.orders if not o.is_completed]

    def get_pending_orders(self) -> List[DIYOrder]:
        """Get all unassigned orders."""
        return [o for o in self.orders if o.assigned_bot_id is None and not o.is_completed]

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
    pan_placed: bool = False
    plate_placed: bool = False

    def is_idle(self) -> bool:
        return self.current_order is None

    def assign_order(self, order: DIYOrder) -> None:
        """Assign a new order to this bot."""
        self.current_order = order
        self.current_session = None
        self.plate_placed = False
        self.pan_placed = False

    def finish_order(self) -> None:
        """Mark order as complete and reset to idle."""
        if self.current_order:
            self.current_order.is_completed = True
        self.current_order = None
        self.current_session = None
        self.plate_placed = False


class BotPlayer:
    def __init__(self, map_copy):
        self.map = map_copy

        # Tile locations
        self.counter_loc: Optional[Tuple[int, int]] = None
        self.cooker_loc: Optional[Tuple[int, int]] = None
        self.shop_loc: Optional[Tuple[int, int]] = None
        self.submit_loc: Optional[Tuple[int, int]] = None
        self.trash_loc: Optional[Tuple[int, int]] = None

        # Scheduler
        self.scheduler = Scheduler()
        self.scheduler_initialized = False

        # Workers for each bot
        self.workers: Dict[int, BotWorker] = {}

        # Route calculator for path reservation
        self.route_calc: Optional[RouteCalculator] = None

    def _init_route_calculator(self, controller: RobotController) -> None:
        """Initialize or update the route calculator with current map."""
        m = controller.get_map(controller.get_team())

        def is_walkable(x: int, y: int) -> bool:
            return m.is_tile_walkable(x, y)

        if self.route_calc is None:
            self.route_calc = RouteCalculator(m.width, m.height, is_walkable)
        else:
            self.route_calc.update_map(m.width, m.height, is_walkable)

    def _find_adjacent_target(self, target_x: int, target_y: int,
                               bot_x: int, bot_y: int) -> Tuple[int, int]:
        """Find the best adjacent cell to a target tile."""
        # If already adjacent, return current position
        if max(abs(bot_x - target_x), abs(bot_y - target_y)) <= 1:
            return (bot_x, bot_y)

        # Find all adjacent walkable cells and pick the closest to bot
        best_pos = (target_x, target_y)  # Fallback
        best_dist = 9999
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ax, ay = target_x + dx, target_y + dy
                if self.route_calc and self.route_calc._in_bounds(ax, ay):
                    if self.route_calc._is_walkable(ax, ay):
                        dist = max(abs(bot_x - ax), abs(bot_y - ay))
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = (ax, ay)
        return best_pos

    def move_towards(self, controller: RobotController, bot_id: int,
                     target_x: int, target_y: int) -> bool:
        """
        Move bot towards target using RouteCalculator.
        Returns True if adjacent to target.
        """
        bot_state = controller.get_bot_state(bot_id)
        bx, by = bot_state['x'], bot_state['y']

        # Check if already adjacent to target
        if max(abs(bx - target_x), abs(by - target_y)) <= 1:
            # Release any existing reservation since we've arrived
            if self.route_calc:
                self.route_calc.release_path(bot_id)
            return True

        if self.route_calc is None:
            return False

        # Find an adjacent walkable cell as destination
        dest = self._find_adjacent_target(target_x, target_y, bx, by)

        # Check if we already have a reservation
        if self.route_calc.has_reservation(bot_id):
            step = self.route_calc.get_next_step(bot_id)
            if step:
                dx, dy = step
                controller.move(bot_id, dx, dy)
            else:
                # No more steps - release and re-plan
                self.route_calc.release_path(bot_id)
            return False

        # Try to reserve a new path
        success, path = self.route_calc.try_reserve_path(bot_id, (bx, by), dest)
        if success and path:
            step = self.route_calc.get_next_step(bot_id)
            if step:
                dx, dy = step
                controller.move(bot_id, dx, dy)
        return False

    def find_nearest_tile(self, controller: RobotController, bot_x: int,
                          bot_y: int, tile_name: str) -> Optional[Tuple[int, int]]:
        best_dist = 9999
        best_pos = None
        m = controller.get_map(controller.get_team())
        for x in range(m.width):
            for y in range(m.height):
                tile = m.tiles[x][y]
                if tile.tile_name == tile_name:
                    dist = max(abs(bot_x - x), abs(bot_y - y))
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (x, y)
        return best_pos

    def initialize_locations(self, controller: RobotController, bx: int, by: int) -> bool:
        """Initialize all tile locations. Returns True if successful."""
        if self.counter_loc is None:
            self.counter_loc = self.find_nearest_tile(controller, bx, by, "COUNTER")
        if self.cooker_loc is None:
            self.cooker_loc = self.find_nearest_tile(controller, bx, by, "COOKER")
        if self.shop_loc is None:
            self.shop_loc = self.find_nearest_tile(controller, bx, by, "SHOP")
        if self.submit_loc is None:
            self.submit_loc = self.find_nearest_tile(controller, bx, by, "SUBMIT")
        if self.trash_loc is None:
            self.trash_loc = self.find_nearest_tile(controller, bx, by, "TRASH")

        if all([self.counter_loc, self.cooker_loc, self.shop_loc, self.submit_loc]):
            if not self.scheduler_initialized:
                self.scheduler.set_locations(
                    shop=self.shop_loc,
                    counter=self.counter_loc,
                    cooker=self.cooker_loc,
                    submit=self.submit_loc
                )
                self.scheduler_initialized = True
            return True
        return False

    def execute_action(self, controller: RobotController, bot_id: int,
                       action: Action) -> bool:
        """
        Execute a single action. Returns True if action completed.
        """
        bot_info = controller.get_bot_state(bot_id)
        tx, ty = action.target_coord if action.target_coord else (0, 0)

        if action.action_type == ActionType.BUY:
            if bot_info.get('holding'):
                return False  # Can't buy if holding something
            if self.move_towards(controller, bot_id, tx, ty):
                if action.food_type:
                    cost = action.food_type.buy_cost
                    if controller.get_team_money(controller.get_team()) >= cost:
                        return controller.buy(bot_id, action.food_type, tx, ty)
                elif action.item_type:
                    cost = action.item_type.buy_cost
                    if controller.get_team_money(controller.get_team()) >= cost:
                        return controller.buy(bot_id, action.item_type, tx, ty)
            return False

        elif action.action_type == ActionType.PLACE:
            if not bot_info.get('holding'):
                return False  # Nothing to place
            if self.move_towards(controller, bot_id, tx, ty):
                return controller.place(bot_id, tx, ty)
            return False

        elif action.action_type == ActionType.PICKUP:
            if bot_info.get('holding'):
                return False  # Already holding something
            if self.move_towards(controller, bot_id, tx, ty):
                return controller.pickup(bot_id, tx, ty)
            return False

        elif action.action_type == ActionType.CHOP:
            if bot_info.get('holding'):
                return False  # Must have empty hands to chop
            if self.move_towards(controller, bot_id, tx, ty):
                return controller.chop(bot_id, tx, ty)
            return False

        elif action.action_type == ActionType.TAKE_FROM_PAN:
            if bot_info.get('holding'):
                return False
            if self.move_towards(controller, bot_id, tx, ty):
                return controller.take_from_pan(bot_id, tx, ty)
            return False

        elif action.action_type == ActionType.ADD_TO_PLATE:
            if not bot_info.get('holding'):
                return False  # Need food to add
            if self.move_towards(controller, bot_id, tx, ty):
                return controller.add_food_to_plate(bot_id, tx, ty)
            return False

        elif action.action_type == ActionType.SUBMIT:
            if self.move_towards(controller, bot_id, tx, ty):
                return controller.submit(bot_id, tx, ty)
            return False

        return False

    def execute_session(self, controller: RobotController, bot_id: int,
                        session: Session) -> bool:
        """
        Execute the current action in a session.
        Returns True if the entire session is completed.
        """
        action = session.get_current_action()
        if action is None:
            return True  # Session complete

        if self.execute_action(controller, bot_id, action):
            return session.advance_action()
        return False

    def ensure_pan_on_cooker(self, controller: RobotController, bot_id: int,
                             worker: BotWorker) -> bool:
        """Ensure pan is on cooker. Returns True if ready."""
        kx, ky = self.cooker_loc
        tile = controller.get_tile(controller.get_team(), kx, ky)

        if tile and isinstance(tile.item, Pan):
            worker.pan_placed = True
            return True

        # Need to place pan - create a one-time session
        bot_info = controller.get_bot_state(bot_id)
        holding = bot_info.get('holding')

        if holding and holding.get('type') == 'Pan':
            if self.move_towards(controller, bot_id, kx, ky):
                if controller.place(bot_id, kx, ky):
                    worker.pan_placed = True
                    return True
        else:
            sx, sy = self.shop_loc
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(controller.get_team()) >= ShopCosts.PAN.buy_cost:
                    controller.buy(bot_id, ShopCosts.PAN, sx, sy)
        return False

    def ensure_plate_on_counter(self, controller: RobotController, bot_id: int,
                                worker: BotWorker) -> bool:
        """Ensure plate is on counter. Returns True if ready."""
        cx, cy = self.counter_loc
        tile = controller.get_tile(controller.get_team(), cx, cy)

        if tile and isinstance(tile.item, Plate) and not tile.item.dirty:
            worker.plate_placed = True
            return True

        bot_info = controller.get_bot_state(bot_id)
        holding = bot_info.get('holding')

        if holding and holding.get('type') == 'Plate':
            if self.move_towards(controller, bot_id, cx, cy):
                if controller.place(bot_id, cx, cy):
                    worker.plate_placed = True
                    return True
        else:
            sx, sy = self.shop_loc
            if self.move_towards(controller, bot_id, sx, sy):
                if controller.get_team_money(controller.get_team()) >= ShopCosts.PLATE.buy_cost:
                    controller.buy(bot_id, ShopCosts.PLATE, sx, sy)
        return False

    def run_bot(self, controller: RobotController, bot_id: int) -> None:
        """Run a single bot's turn using session-based scheduling."""
        turn = controller.get_turn()

        # Get or create worker
        if bot_id not in self.workers:
            self.workers[bot_id] = BotWorker(bot_id=bot_id)
        worker = self.workers[bot_id]

        # If idle, get a new order
        if worker.is_idle():
            order = self.scheduler.get_next_order()
            if order:
                worker.assign_order(order)
                self.scheduler.assign_order(order.order_id, bot_id)
                foods_str = ", ".join(f.food_name for f in order.required_foods)
                print(f"[Turn {turn}] Bot {bot_id}: Assigned order {order.order_id} ({foods_str})")

        if worker.is_idle():
            print(f"[Turn {turn}] Bot {bot_id}: IDLE (no orders)")
            return

        order = worker.current_order

        # Check if order needs cooking - ensure pan is ready FIRST
        needs_cooking = any(f.can_cook for f in order.required_foods)
        if needs_cooking and not worker.pan_placed:
            print(f"[Turn {turn}] Bot {bot_id}: Ensuring pan on cooker...")
            if not self.ensure_pan_on_cooker(controller, bot_id, worker):
                return

        # Get next available session (priority-based)
        session = order.get_next_session(turn)

        if session:
            # Check if this session needs plate - ensure plate is ready
            needs_plate = session.session_type in [
                SessionType.BUY_AND_PLATE_SIMPLE,
                SessionType.BUY_CHOP_AND_PLATE,
                SessionType.TAKE_AND_PLATE_COOKED,
                SessionType.BUY_AND_PLACE_PLATE
            ]
            if needs_plate and not worker.plate_placed:
                print(f"[Turn {turn}] Bot {bot_id}: Ensuring plate on counter...")
                if not self.ensure_plate_on_counter(controller, bot_id, worker):
                    return

            food = order.get_food_for_session(session)
            food_name = food.food_name if food else "?"
            action = session.get_current_action()
            action_str = f"{action.action_type.name} @ {action.target_coord}" if action else "None"
            print(f"[Turn {turn}] Bot {bot_id}: Session={session.session_type.name}, "
                  f"Food={food_name}, Action={action_str}")

            # Execute current session
            if self.execute_session(controller, bot_id, session):
                print(f"[Turn {turn}] Bot {bot_id}: Session COMPLETED!")

                # Advance the food's session tracking
                if food:
                    food.advance_session()
                    if food.is_prepared:
                        print(f"[Turn {turn}] Bot {bot_id}: {food_name} is fully prepared!")

                # If this was a cooking session, set the cook ready time
                if session.session_type in [SessionType.BUY_CHOP_AND_COOK, SessionType.BUY_AND_COOK]:
                    # Find the TAKE_AND_PLATE_COOKED session for this food
                    if food:
                        for s in food.sessions:
                            if s.session_type == SessionType.TAKE_AND_PLATE_COOKED:
                                s.available_after_turn = turn + GameConstants.COOK_PROGRESS
                                print(f"[Turn {turn}] Bot {bot_id}: {food_name} will be ready at turn {s.available_after_turn}")
                                break
        else:
            # No available sessions - check if all foods are prepared
            if order.all_foods_prepared():
                print(f"[Turn {turn}] Bot {bot_id}: All foods ready, submitting...")
                bot_info = controller.get_bot_state(bot_id)
                holding = bot_info.get('holding')
                cx, cy = self.counter_loc
                ux, uy = self.submit_loc

                if not holding:
                    if self.move_towards(controller, bot_id, cx, cy):
                        controller.pickup(bot_id, cx, cy)
                else:
                    if self.move_towards(controller, bot_id, ux, uy):
                        if controller.submit(bot_id, ux, uy):
                            print(f"[Turn {turn}] Bot {bot_id}: ORDER {order.order_id} SUBMITTED!")
                            self.scheduler.complete_order(order.order_id)
                            worker.finish_order()
            else:
                # Waiting for something (cooking)
                print(f"[Turn {turn}] Bot {bot_id}: Waiting (cooking in progress)...")

    def play_turn(self, controller: RobotController):
        """Main entry point - called each turn."""
        my_bots = controller.get_team_bot_ids(controller.get_team())
        if not my_bots:
            return

        # Initialize/update route calculator for this turn
        self._init_route_calculator(controller)

        # Initialize locations using first bot
        bot_info = controller.get_bot_state(my_bots[0])
        bx, by = bot_info['x'], bot_info['y']

        if not self.initialize_locations(controller, bx, by):
            return

        # Update scheduler with new orders
        self.scheduler.receive_orders(controller, controller.get_turn())

        # Run each bot
        for bot_id in my_bots:
            self.run_bot(controller, bot_id)

        # Advance route reservations for next turn
        if self.route_calc:
            self.route_calc.advance_round()