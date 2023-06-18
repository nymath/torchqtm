from typing import Callable
from torchqtm.types import F
from functools import wraps


def ContextManager(manager, used=True) -> Callable[[F], F]:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if used:
                with manager:
                    func(*args, **kwargs)
            else:
                func(*args, **kwargs)
        return wrapper
    return decorator







