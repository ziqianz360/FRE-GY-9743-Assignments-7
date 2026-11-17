import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Continuous piecewise-linear IFR interpolation for monotone, convex data.

    Convexity requires nondecreasing secant slopes, allowing floating-point
    roundoff. Shape guarantees apply only between the first and last nodes;
    FLAT extrapolation need not preserve convexity beyond that interval.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        axis1 = np.array(axis1, dtype=float, copy=True)
        values = np.array(values, dtype=float, copy=True)
        if axis1.ndim != 1 or values.ndim != 1:
            raise ValueError("Node times and IFR values must be one-dimensional.")
        if len(axis1) == 0 or len(axis1) != len(values):
            raise ValueError("Node times and IFR values must have equal, nonzero length.")
        if not np.all(np.isfinite(axis1)) or not np.all(np.isfinite(values)):
            raise ValueError("Node times and IFR values must be finite.")
        if not np.all(np.diff(axis1) > 0):
            raise ValueError("Node times must be strictly increasing.")
        if extrapolation_method != ExtrapMethod.FLAT:
            raise ValueError("Monotone convex interpolation requires FLAT extrapolation.")

        changes = np.diff(values)
        if not (np.all(changes >= 0) or np.all(changes <= 0)):
            raise ValueError("IFR values must be monotone to preserve all supplied points.")
        slopes = changes / np.diff(axis1)
        if not np.all(np.isfinite(slopes)):
            raise ValueError("IFR secant slopes must be finite.")
        if len(slopes) > 1:
            tolerance = 64 * np.finfo(float).eps * np.max(np.abs(slopes))
            if np.any(np.diff(slopes) < -tolerance):
                raise ValueError("Convex IFR data require nondecreasing secant slopes.")

        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        self.slopes_ = slopes
        areas = 0.5 * (values[:-1] + values[1:]) * np.diff(axis1)
        self.integrated_values_ = np.concatenate(([0.0], np.cumsum(areas)))

    def interpolate(self, x: float) -> float:
        if not np.isfinite(x):
            raise ValueError("Interpolation time must be finite.")
        if x <= self.axis1_[0]:
            return float(self.values_[0])
        if x >= self.axis1_[-1]:
            return float(self.values_[-1])
        index = int(np.searchsorted(self.axis1_, x, side='right')) - 1
        return float(self.values_[index] + self.slopes_[index] * (x - self.axis1_[index]))

    def integrate(self, start_x: float, end_x: float) -> float:
        if not np.isfinite(start_x) or not np.isfinite(end_x):
            raise ValueError("Integration limits must be finite.")
        return self._primitive(end_x) - self._primitive(start_x)

    def _primitive(self, x: float) -> float:
        """Integral from the first node to x, including flat extrapolation."""
        if x <= self.axis1_[0]:
            return float(self.values_[0] * (x - self.axis1_[0]))
        if x >= self.axis1_[-1]:
            return float(self.integrated_values_[-1] + self.values_[-1] * (x - self.axis1_[-1]))
        index = int(np.searchsorted(self.axis1_, x, side='right')) - 1
        dx = x - self.axis1_[index]
        return float(self.integrated_values_[index] + self.values_[index] * dx
                     + 0.5 * self.slopes_[index] * dx ** 2)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
