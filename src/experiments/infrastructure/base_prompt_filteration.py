from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable, List, Optional, final

from src.core.types import TPromptOriginalIndex
from src.utils.jsonable import JSONAble
from src.utils.types_utils import BaseParams

if TYPE_CHECKING:
    from src.experiments.infrastructure.base_runner import BaseRunner, TDependencies


@dataclass(frozen=True)
class BasePromptFilteration(BaseParams, JSONAble):
    """Filteration of prompts to run the experiment on."""

    _context: Optional[Any] = field(default=None, init=False)

    @abstractmethod
    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        pass

    @abstractmethod
    def get_dependencies(self) -> TDependencies:
        pass

    @final
    def contextualize(self, context: Any):
        if self._context is not None:
            return self
        new_self = self.modify()
        object.__setattr__(new_self, "_context", context)
        return new_self

    @final
    @property
    def has_context(self) -> bool:
        return self._context is not None

    def uncomputed_dependencies(self) -> list[BaseRunner]:
        def rec_uncomputed_dependencies(
            dependencies: TDependencies,
        ) -> list[BaseRunner]:
            from src.experiments.infrastructure.base_runner import BaseRunner

            res = []
            for k, v in dependencies.items():
                if isinstance(v, BaseRunner):
                    if not v.is_computed():
                        res.append(v)
                else:
                    res.extend(rec_uncomputed_dependencies(v))
            return res

        return rec_uncomputed_dependencies(self.get_dependencies())

    def dependencies_are_computed(self) -> bool:
        return len(self.uncomputed_dependencies()) == 0

    @abstractmethod
    def display_name(self) -> str:
        pass

    def __and__(self, other: BasePromptFilteration) -> LogicalPromptFilteration:
        """Support for filter1 & filter2 syntax (AND operation)"""
        return LogicalPromptFilteration.create_and([self, other])

    def __or__(self, other: BasePromptFilteration) -> LogicalPromptFilteration:
        """Support for filter1 | filter2 syntax (OR operation)"""
        return LogicalPromptFilteration.create_or([self, other])

    def __invert__(self) -> LogicalPromptFilteration:
        """Support for ~filter syntax (NOT operation)"""
        return LogicalPromptFilteration.create_not(self)


@dataclass(frozen=True)
class SelectivePromptFilteration(BasePromptFilteration):
    prompt_ids: tuple[TPromptOriginalIndex, ...]

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        return list(self.prompt_ids)

    def get_dependencies(self) -> TDependencies:
        return {}

    def display_name(self) -> str:
        amount = len(self.prompt_ids)
        if amount > 5:
            return f"Selective ({amount})"
        else:
            return f"Selective ({', '.join(str(prompt_id) for prompt_id in self.prompt_ids)})"


class LogicalOperationType(StrEnum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass(frozen=True)
class ProxyPromptFilteration(BasePromptFilteration, ABC):
    @abstractmethod
    def _get_prompt_ids(
        self, get_prompt_ids: Callable[[BasePromptFilteration], list[TPromptOriginalIndex]]
    ) -> list[TPromptOriginalIndex]:
        pass

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:
        def func(filteration: BasePromptFilteration) -> list[TPromptOriginalIndex]:
            if self._context is not None:
                return filteration.contextualize(self._context).get_prompt_ids()
            return filteration.get_prompt_ids()

        return self._get_prompt_ids(func)


@dataclass(frozen=True)
class LogicalPromptFilteration(ProxyPromptFilteration):
    """Filteration that supports logical operations (AND, OR, NOT) between other filterations."""

    operation_type: LogicalOperationType
    operands: tuple[BasePromptFilteration, ...] = field(default_factory=tuple)
    # For NOT operations, we need to know the universe of prompt IDs
    universe: Optional[BasePromptFilteration] = None

    @classmethod
    def create_and(cls, filterations: List[BasePromptFilteration]):
        """Create an AND (intersection) operation between filterations."""
        return cls(operation_type=LogicalOperationType.AND, operands=tuple(filterations))

    @classmethod
    def create_or(cls, filterations: List[BasePromptFilteration]):
        """Create an OR (union) operation between filterations."""
        return cls(operation_type=LogicalOperationType.OR, operands=tuple(filterations))

    @classmethod
    def create_not(cls, filteration: BasePromptFilteration, universe: Optional[BasePromptFilteration] = None):
        """Create a NOT operation for a filteration.

        Args:
            filteration: The filteration to negate
            universe: Optional universe of prompt IDs to negate against
        """
        return cls(operation_type=LogicalOperationType.NOT, operands=(filteration,), universe=universe)

    def with_universe(self, universe: BasePromptFilteration):
        """Set the universe for NOT operations."""
        if self.operation_type != LogicalOperationType.NOT:
            return self

        return self.modify(universe=universe)

    def add_operand(self, filteration: BasePromptFilteration):
        """Add another filteration to this logical operation."""
        if self.operation_type == LogicalOperationType.NOT:
            # NOT operations only support one operand
            return self

        # If adding the same type of LogicalPromptFilteration, flatten the structure
        if isinstance(filteration, LogicalPromptFilteration) and filteration.operation_type == self.operation_type:
            return self.modify(operands=self.operands + filteration.operands)
        else:
            return self.modify(operands=self.operands + (filteration,))

    def and_with(self, filteration: BasePromptFilteration):
        """Create a new AND operation with this filteration and another."""
        if self.operation_type == LogicalOperationType.AND:
            return self.add_operand(filteration)
        return LogicalPromptFilteration.create_and([self, filteration])

    def or_with(self, filteration: BasePromptFilteration):
        """Create a new OR operation with this filteration and another."""
        if self.operation_type == LogicalOperationType.OR:
            return self.add_operand(filteration)
        return LogicalPromptFilteration.create_or([self, filteration])

    def not_op(self, universe: Optional[BasePromptFilteration] = None):
        """Create a new NOT operation of this filteration."""
        return LogicalPromptFilteration.create_not(self, universe)

    def _get_prompt_ids(
        self, get_prompt_ids: Callable[[BasePromptFilteration], list[TPromptOriginalIndex]]
    ) -> list[TPromptOriginalIndex]:
        if self.operation_type == LogicalOperationType.AND:
            if not self.operands:
                return []
            # Intersection operation (AND)
            result = set(get_prompt_ids(self.operands[0]))
            for filteration in self.operands[1:]:
                result.intersection_update(set(get_prompt_ids(filteration)))
            return list(result)

        elif self.operation_type == LogicalOperationType.OR:
            # Union operation (OR)
            result = set()
            for filteration in self.operands:
                result.update(get_prompt_ids(filteration))
            return list(result)

        elif self.operation_type == LogicalOperationType.NOT:
            # NOT operation requires a reference set to negate against
            if not self.operands:
                return []

            filter_set = set(get_prompt_ids(self.operands[0]))

            # If we have a universe, use it; otherwise return an empty set
            if self.universe:
                universe_set = set(get_prompt_ids(self.universe))
                return list(universe_set - filter_set)

            # Without a universe specified, we can't properly implement NOT
            return []

        return []

    def get_prompt_ids(self) -> list[TPromptOriginalIndex]:  # type: ignore
        return super().get_prompt_ids()

    def get_dependencies(self) -> TDependencies:
        dependencies = {}
        for filteration in self.operands:
            deps = filteration.get_dependencies()
            if deps:
                dependencies[filteration] = deps

        # Add dependencies from universe if present
        if self.universe:
            universe_deps = self.universe.get_dependencies()
            if universe_deps:
                dependencies["universe"] = universe_deps

        return dependencies

    def display_name(self):
        return f"{self.operation_type} on {len(self.operands)} operands"


@dataclass(frozen=True)
class SamplePromptFilteration(ProxyPromptFilteration):
    base_prompt_filteration: BasePromptFilteration
    sample_size: int
    seed: int

    def _get_prompt_ids(
        self, get_prompt_ids: Callable[[BasePromptFilteration], list[TPromptOriginalIndex]]
    ) -> list[TPromptOriginalIndex]:
        random.seed(self.seed)
        prompt_ids = get_prompt_ids(self.base_prompt_filteration)
        if len(prompt_ids) <= self.sample_size:
            return prompt_ids
        return random.sample(prompt_ids, self.sample_size)

    def get_dependencies(self) -> TDependencies:
        return self.base_prompt_filteration.get_dependencies()

    def display_name(self) -> str:
        return f"Sample ({self.sample_size})"
