from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import gcd, prod
from operator import index
from typing import Any, Iterable

from ._core import M
from .capacity import SignedCapacityPlan


def _require_integer(value: Any, *, name: str) -> int:
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _require_positive_integer(value: Any, *, name: str) -> int:
    integer = _require_integer(value, name=name)
    if integer <= 0:
        raise ValueError(f"{name} must be > 0")
    return integer


def _require_exponent(value: Any, *, name: str) -> int:
    exponent = _require_integer(value, name=name)
    if exponent < 2:
        raise ValueError(f"{name} must be >= 2")
    return exponent


@dataclass(frozen=True, slots=True, order=True)
class MersenneRailCandidate:
    """One CRT candidate modulus of the form ``2**exponent - 1``.

    The modulus is not required to be prime. CRT needs pairwise coprimality, not
    primality. Composite candidates can therefore be lawful for addition and
    multiplication while having stricter division invertibility conditions.
    """

    exponent: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exponent",
            _require_exponent(self.exponent, name="exponent"),
        )

    @property
    def modulus(self) -> int:
        return (1 << self.exponent) - 1

    @property
    def storage_bits(self) -> int:
        return self.exponent

    @property
    def coprime_with_current_engine(self) -> bool:
        return gcd(self.modulus, int(M)) == 1


@dataclass(frozen=True, slots=True)
class MersenneRailSetPlan:
    """Exact capacity receipt for one proposed Mersenne-form rail set."""

    required_additional_product: int
    rails: tuple[MersenneRailCandidate, ...]

    def __post_init__(self) -> None:
        required = _require_positive_integer(
            self.required_additional_product,
            name="required_additional_product",
        )
        raw_rails = tuple(self.rails)
        if len({rail.exponent for rail in raw_rails}) != len(raw_rails):
            raise ValueError("rail exponents must be distinct")

        ordered = tuple(sorted(raw_rails))
        running = int(M)
        for position, rail in enumerate(ordered):
            if not isinstance(rail, MersenneRailCandidate):
                raise TypeError(
                    f"rails[{position}] must be a MersenneRailCandidate"
                )
            common = gcd(running, rail.modulus)
            if common != 1:
                raise ValueError(
                    f"rail exponent {rail.exponent} gives modulus "
                    f"{rail.modulus}, which is not coprime with the running "
                    f"CRT product; gcd={common}"
                )
            running *= rail.modulus

        object.__setattr__(self, "required_additional_product", required)
        object.__setattr__(self, "rails", ordered)

    @property
    def exponents(self) -> tuple[int, ...]:
        return tuple(rail.exponent for rail in self.rails)

    @property
    def moduli(self) -> tuple[int, ...]:
        return tuple(rail.modulus for rail in self.rails)

    @property
    def rail_count(self) -> int:
        return len(self.rails)

    @property
    def additional_product(self) -> int:
        return prod(self.moduli, start=1)

    @property
    def additional_product_bits(self) -> int:
        return self.additional_product.bit_length()

    @property
    def total_storage_bits(self) -> int:
        return sum(self.exponents)

    @property
    def exponent_span(self) -> int:
        if not self.rails:
            return 0
        return self.exponents[-1] - self.exponents[0]

    @property
    def sufficient(self) -> bool:
        return self.additional_product >= self.required_additional_product

    @property
    def excess_product(self) -> int:
        return max(0, self.additional_product - self.required_additional_product)

    @property
    def expanded_modulus(self) -> int:
        return int(M) * self.additional_product

    @property
    def capacity_multiple_numerator(self) -> int:
        """Exact numerator of ``additional_product / required_product``."""

        return self.additional_product

    @property
    def capacity_multiple_denominator(self) -> int:
        return self.required_additional_product

    def require_sufficient(self) -> MersenneRailSetPlan:
        if not self.sufficient:
            raise OverflowError(
                "proposed rail set is insufficient: additional product "
                f"{self.additional_product} is smaller than required product "
                f"{self.required_additional_product}"
            )
        return self


@dataclass(frozen=True, slots=True)
class MersenneRailSearchResult:
    """Exact enumeration result under an exponent and rail-count ceiling."""

    required_additional_product: int
    min_exponent: int
    max_exponent: int
    max_rails: int
    candidates: tuple[MersenneRailCandidate, ...]
    solutions: tuple[MersenneRailSetPlan, ...]

    @property
    def minimum_rail_count(self) -> int | None:
        if not self.solutions:
            return None
        return min(plan.rail_count for plan in self.solutions)

    @property
    def smallest_product_plan(self) -> MersenneRailSetPlan | None:
        if not self.solutions:
            return None
        return min(
            self.solutions,
            key=lambda plan: (
                plan.additional_product,
                plan.exponent_span,
                plan.exponents,
            ),
        )

    @property
    def most_balanced_plan(self) -> MersenneRailSetPlan | None:
        if not self.solutions:
            return None
        return min(
            self.solutions,
            key=lambda plan: (
                plan.exponent_span,
                plan.additional_product,
                plan.exponents,
            ),
        )


def search_mersenne_rail_sets(
    required_additional_product: int,
    *,
    min_exponent: int = 2,
    max_exponent: int = 31,
    max_rails: int = 6,
    minimal_rail_count_only: bool = True,
    limit: int | None = None,
) -> MersenneRailSearchResult:
    """Enumerate lawful Mersenne-form CRT rail sets exactly.

    Candidate moduli are ``2**exponent - 1``. They are filtered for
    coprimality with the current four-rail modulus product, then combinations
    are checked incrementally for pairwise coprimality and sufficient product.

    Results are ordered by rail count, added product, exponent span, and the
    exponent tuple. No speed score is assigned: mathematical capacity and
    hardware efficiency are separate questions.
    """

    required = _require_positive_integer(
        required_additional_product,
        name="required_additional_product",
    )
    minimum = _require_exponent(min_exponent, name="min_exponent")
    maximum = _require_exponent(max_exponent, name="max_exponent")
    if maximum < minimum:
        raise ValueError("max_exponent must be >= min_exponent")

    rail_ceiling = _require_positive_integer(max_rails, name="max_rails")
    if limit is not None:
        limit = _require_positive_integer(limit, name="limit")

    candidates = tuple(
        candidate
        for candidate in (
            MersenneRailCandidate(exponent)
            for exponent in range(minimum, maximum + 1)
        )
        if candidate.coprime_with_current_engine
    )

    solutions: list[MersenneRailSetPlan] = []
    for rail_count in range(1, min(rail_ceiling, len(candidates)) + 1):
        found_at_this_count: list[MersenneRailSetPlan] = []
        for rail_tuple in combinations(candidates, rail_count):
            running = int(M)
            lawful = True
            for rail in rail_tuple:
                if gcd(running, rail.modulus) != 1:
                    lawful = False
                    break
                running *= rail.modulus
            if not lawful:
                continue

            plan = MersenneRailSetPlan(required, rail_tuple)
            if plan.sufficient:
                found_at_this_count.append(plan)

        found_at_this_count.sort(
            key=lambda plan: (
                plan.additional_product,
                plan.exponent_span,
                plan.exponents,
            )
        )
        solutions.extend(found_at_this_count)
        if found_at_this_count and minimal_rail_count_only:
            break

    solutions.sort(
        key=lambda plan: (
            plan.rail_count,
            plan.additional_product,
            plan.exponent_span,
            plan.exponents,
        )
    )
    if limit is not None:
        solutions = solutions[:limit]

    return MersenneRailSearchResult(
        required_additional_product=required,
        min_exponent=minimum,
        max_exponent=maximum,
        max_rails=rail_ceiling,
        candidates=candidates,
        solutions=tuple(solutions),
    )


def search_mersenne_rails_for_capacity(
    capacity: SignedCapacityPlan,
    **kwargs: Any,
) -> MersenneRailSearchResult:
    """Search rails for the deficit recorded by a signed capacity receipt."""

    if not isinstance(capacity, SignedCapacityPlan):
        raise TypeError("capacity must be a SignedCapacityPlan")
    return search_mersenne_rail_sets(
        capacity.minimum_additional_product_factor,
        **kwargs,
    )


__all__ = [
    "MersenneRailCandidate",
    "MersenneRailSearchResult",
    "MersenneRailSetPlan",
    "search_mersenne_rail_sets",
    "search_mersenne_rails_for_capacity",
]
