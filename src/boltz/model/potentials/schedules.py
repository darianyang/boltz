import math
from abc import ABC

class ParameterSchedule(ABC):
    def compute(self, t):
        raise NotImplementedError

class ExponentialInterpolation(ParameterSchedule):
    def __init__(self, start, end, alpha):
        self.start = start
        self.end = end
        self.alpha = alpha

    def compute(self, t):
        if self.alpha != 0:
            return self.start + (self.end - self.start) * (math.exp(self.alpha * t) - 1) / (math.exp(self.alpha) - 1)
        else:
            return self.start + (self.end - self.start) * t

class RepeatedExponentialInterpolation(ParameterSchedule):
    def __init__(self, start, end, alpha, n_repeats):
        self.start = start
        self.end = end
        self.alpha = alpha
        self.n_repeats = n_repeats

    def compute(self, t):
        # Map t to the current interval [0, 1/n_repeats, 2/n_repeats, ...]
        t_local = (t * self.n_repeats) % 1.0
        
        # note t goes from 1 --> 0
        # if t is in the beginning interval of 1-->0, return zeros (warm up)
        # or if t == 1, return start/0.0 (prevent end spikes)
        if t > (1 - 1/self.n_repeats) or t == 1.0:
            return self.start

        # Apply exponential interpolation within the local interval
        if self.alpha != 0:
            return self.start + (self.end - self.start) * (math.exp(self.alpha * t_local) - 1) / (math.exp(self.alpha) - 1)
        else:
            return self.start + (self.end - self.start) * t_local

class PiecewiseStepFunction(ParameterSchedule):
    def __init__(self, thresholds, values):
        self.thresholds = thresholds
        self.values = values

    def compute(self, t):
        assert len(self.thresholds) > 0
        assert len(self.values) == len(self.thresholds) + 1

        idx = 0
        while idx < len(self.thresholds) and t > self.thresholds[idx]:
            idx += 1
        return self.values[idx]