from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import threading

logger = logging.getLogger(__name__)


class HookPoint(Enum):
    GENERATION_START = "generation_start"
    LM_PHASE_START = "lm_phase_start"
    LM_PHASE_END = "lm_phase_end"
    DIT_PHASE_START = "dit_phase_start"
    DIT_PHASE_END = "dit_phase_end"
    GENERATION_END = "generation_end"
    GENERATION_ERROR = "generation_error"


@dataclass
class HookContext:
    hook_point: HookPoint
    data: Dict[str, Any]


HookCallback = Callable[[HookContext], None]


class HookRegistry:
    _instance: Optional["HookRegistry"] = None
    _creation_lock = threading.Lock()
    _hooks: Dict[HookPoint, List[HookCallback]]

    def __new__(cls) -> "HookRegistry":
        with cls._creation_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._hooks = {p: [] for p in HookPoint}
            return cls._instance

    def register(self, hook_point: HookPoint, callback: HookCallback) -> None:
        self._hooks[hook_point].append(callback)

    def unregister(self, hook_point: HookPoint, callback: HookCallback) -> None:
        if callback in self._hooks[hook_point]:
            self._hooks[hook_point].remove(callback)

    def clear(self, hook_point: Optional[HookPoint] = None) -> None:
        if hook_point:
            self._hooks[hook_point] = []
        else:
            self._hooks = {p: [] for p in HookPoint}

    def fire(self, hook_point: HookPoint, data: Optional[Dict[str, Any]] = None) -> None:
        ctx = HookContext(hook_point=hook_point, data=data or {})
        for cb in list(self._hooks[hook_point]):
            try:
                cb(ctx)
            except Exception as e:
                logger.warning(f"Hook callback error at {hook_point.value}: {e}")


hooks = HookRegistry()
