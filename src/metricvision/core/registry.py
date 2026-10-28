"""A tiny generic registry so components can be selected by name from config files."""

from __future__ import annotations  # Postponed evaluation of type hints.

from collections.abc import Callable  # Type of the decorator returned by register().
from typing import Any, Generic, TypeVar  # Generic registry typing.

T = TypeVar("T")  # The base class this registry holds (e.g. FeatureMatcher).


class Registry(Generic[T]):
    """Maps string names to subclasses of a common base class.

    Example:
        >>> MATCHERS: Registry[FeatureMatcher] = Registry("matcher")
        >>> @MATCHERS.register("orb")
        ... class OrbMatcher(FeatureMatcher): ...
        >>> matcher = MATCHERS.create("orb", max_keypoints=2000)
    """

    def __init__(self, kind: str) -> None:
        """Create an empty registry.

        Args:
            kind: Human-readable component kind, used in error messages.
        """
        self._kind = kind  # E.g. "matcher", "dataset".
        self._entries: dict[str, type[T]] = {}  # Name -> class mapping.

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a class decorator that registers the decorated class under ``name``.

        Args:
            name: Key used to select the class from configs and the CLI.

        Returns:
            A decorator that registers the class and returns it unchanged.
        """

        def decorator(cls: type[T]) -> type[T]:
            """Register ``cls`` and hand it back unchanged.

            Args:
                cls: The class being decorated.

            Returns:
                The same class, so decoration has no other side effects.

            Raises:
                KeyError: If the name is already taken.
            """
            if name in self._entries:  # Duplicate names would silently shadow each other.
                raise KeyError(f"{self._kind} '{name}' is already registered")  # Fail loudly instead.
            self._entries[name] = cls  # Store the mapping.
            return cls  # Decorators must return the class.

        return decorator  # Used as @REGISTRY.register("name").

    def get(self, name: str) -> type[T]:
        """Look up a registered class by name.

        Args:
            name: Registered key.

        Returns:
            The registered class.

        Raises:
            KeyError: If no class is registered under ``name``.
        """
        if name not in self._entries:  # Unknown key.
            available = ", ".join(self.names()) or "<none>"  # List options to make typos easy to fix.
            raise KeyError(f"Unknown {self._kind} '{name}'. Available: {available}")  # Helpful error.
        return self._entries[name]  # Registered class.

    def create(self, name: str, **kwargs: Any) -> T:
        """Instantiate a registered class.

        Args:
            name: Registered key.
            **kwargs: Constructor arguments forwarded to the class.

        Returns:
            A new instance of the registered class.
        """
        return self.get(name)(**kwargs)  # Look up, then construct.

    def names(self) -> list[str]:
        """Return all registered names in alphabetical order.

        Returns:
            Sorted list of registered keys.
        """
        return sorted(self._entries)  # Sorted for stable CLI output.
