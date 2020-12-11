# /bin/env python3
# -*- coding: utf-8 -*-
# trace.py --- Data type for iterative algorithm

# Copyright (C) 2020 François Orieux <francois.orieux@universite-paris-saclay.fr>

# Everyone is permitted to copy and distribute verbatim or modified
# copies of this license document, and changing it is allowed as long
# as the name is changed.

#            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
#   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

#  0. You just DO WHAT THE FUCK YOU WANT TO.


# Commentary:

"""
References
----------
"""

# code:

from __future__ import division  # Then 10/3 provide 3.3333 instead of 3

import abc
import collections
import functools
import math
import tempfile
import time
import warnings

import h5py
import numpy as np

try:
    import tqdm
except:
    pass

try:
    import notify2
except ImportError:
    pass
else:
    if not notify2.initted:
        notify2.init("Trace active")

__author__ = "François Orieux"
__copyright__ = "2018-2020 F. Orieux <orieux@l2s.centralesupelec.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@l2s.centralesupelec.fr"
__status__ = "early alpha"
__url__ = ""
__keywords__ = "tools, algorithmes"


#%% Traces
class Trace(collections.abc.Sequence):
    """The main data type of the Package"""

    def __init__(self, init=None, name="", backend=None):
        self.name = name
        self.timestamp = [time.time()]
        if backend is None:
            self.backend = ListBackend(init=init)
        self._observers = []

    def append(self, value):
        """Append the value at the end of the trace."""
        self.backend.append(value)
        for obs in self._observers:
            obs.update()

    @property
    def last(self):
        """Return the last element of the trace. Equivalent to self[-1]. If the trace is
        empty, return `None`.

        """
        return self.backend.last

    @last.setter
    def last(self, value):
        """Add an value at the end of the trace. Equivalent to the append method of the
        list type.

        """
        self.backend.append(value)
        self.timestamp.append(time.time())

    def __ilshift__(self, value):
        """Use <<= as a affectation or Trace ← ('gets') meaning"""
        self.backend.append(value)
        return self

    @property
    def time(self):
        """Return effective time of controlled iteration"""
        return np.array(self.timestamp) - self.timestamp[0]

    def register(self, obs):
        self._observers.append(obs)

    #%% Arithmetic
    def __pos__(self):
        return +self.last

    def __neg__(self):
        return -self.last

    def __abs__(self):
        return abs(self.last)

    def __round__(self, n):
        return round(self.last)

    def __floor__(self):
        return math.floor(self.last)

    def __ceil__(self):
        return math.ceil(self.last)

    def __trunc__(self):
        return math.trunc(self.last)

    def __add__(self, value):
        return self.last + value

    def __sub__(self, value):
        return self.last - value

    def __mul__(self, value):
        return self.last * value

    def __floordiv__(self, value):
        return self.last // value

    def __truediv__(self, value):
        return self.last / value

    def __mod__(self, value):
        return self.last % value

    def __divmod__(self, value):
        return divmod(self.last, value)

    def __pow__(self, value):
        return pow(self.last, value)

    def __radd__(self, value):
        return value + self.last

    def __rsub__(self, value):
        return value - self.last

    def __rmul__(self, value):
        return value * self.last

    def __rfloordiv__(self, value):
        return value // self.last

    def __rtruediv__(self, value):
        return value / self.last

    def __rmod__(self, value):
        return value % self.last

    def __rdivmod__(self, value):
        return divmod(value, self.last)

    def __rpow__(self, value):
        return pow(value, self.last)

    def sum(self, start=0):
        """Return the sum of the trace, starting from `start`."""
        return self.backend.sum(start)

    def mean(self, start=0):
        """Return the mean of the trace, starting from `start`."""
        return self.backend.mean(start)

    def var(self, start=0):
        """Return the element wise variance of the trace, starting from `start`."""
        return self.backend.var(start)

    def std(self, start=0):
        """Return the element wise standard deviation of the trace, starting from
        `start`."""
        return self.backend.std(start)

    @property
    def delta(self):
        """Return |self[-2] - self[-1]|^2 / |self[-1]|^2 if defined, else ∞."""
        return self.backend.delta

    @property
    def shape(self):
        """Return the shape of values"""
        return self.backend.last.shape

    @property
    def ndim(self):
        """Return the ndim of values"""
        return len(self.backend.last.shape)

    @property
    def full_shape(self):
        """Return the shape of the trace"""
        return (len(self),) + self.shape

    @property
    def size(self):
        """Return the size of values"""
        return np.prod(self.backend.last.shape)

    def __len__(self):
        """Return the length of the trace"""
        return len(self.backend)

    def __getitem__(self, key):
        return self.backend[key]

    def __repr__(self):
        return "{} ({}) / length: {} × shape: {}".format(
            self.name, type(self).__name__, len(self), self.shape
        )


class StochTrace(Trace):
    """Stochastic version of Trace"""

    def __init__(self, burnin=0, init=None, name="", backend=None):
        super().__init__(init=init, name=name, backend=backend)
        self.burnin = burnin

    @property
    def burned(self):
        """Return `True` is the length of the Trace is higher than `burnin` attribut."""
        if len(self) >= self.burnin:
            return True
        else:
            return False

    def sum(self, start=None):
        """Return the sum of the trace, starting from `start`."""
        return self.backend.sum(start if start is not None else self.burnin)

    def mean(self, start=None):
        """Return the mean of the trace, starting from `start`."""
        return self.backend.mean(start if start is not None else self.burnin)

    def var(self, start=None):
        """Return the element wise variance of the trace, starting from `start`."""
        return self.backend.var(start if start is not None else self.burnin)

    def std(self, start=None):
        """Return the element wise standard deviation of the trace, starting from
        `start`."""
        return self.backend.std(start if start is not None else self.burnin)

    def __repr__(self):
        return "{} ({}) / length: {} [burn-in: {} {}] × shape: {}".format(
            self.name,
            type(self).__name__,
            len(self),
            self.burnin,
            "✓" if self.burned else "⍻",
            self.shape,
        )


class Backend(collections.abc.Sequence):
    """An backend abstract base class

    Backend are in charge of storing the values and reply to some common
    calculus that depends on the storage for efficiency

    Values must be seen as numpy.array.
    """

    @abc.abstractmethod
    def append(self, value):
        """Append a value at the end of the backend"""
        return NotImplemented

    @property
    @abc.abstractmethod
    def last(self):
        """The last value stored in the backend"""
        return NotImplemented

    @abc.abstractmethod
    def sum(self, sum_start=None):
        """Return the sum of values"""
        return NotImplemented

    @abc.abstractmethod
    def mean(self, sum_start=None):
        """Return the mean of values"""
        return NotImplemented

    @abc.abstractmethod
    def var(self, sum_start=None):
        """Return the variance of values"""
        return NotImplemented

    @abc.abstractmethod
    def std(self, sum_start=None):
        """Return the standard deviation of values"""
        return NotImplemented

    def asarray(self):
        """Return the values as numpy array"""
        return np.array([val[np.newaxis] for val in self])

    def as_hdf5_dataset(self, dataset):
        """Return the values as numpy array"""
        dataset = self.asarray()

    def as_hdf5_file(self, filename):
        """Return the values as numpy array"""
        h5file = h5py.File(filename)
        h5file.create_dataset("values", data=self.asarray())
        return h5file

    @property
    def delta(self):
        """Return |self[-2] - self[-1]|^2 / |self[-1]|^2 if defined, else ∞."""
        if len(self) >= 2:
            return np.sum((self[-1] - self[-2]) ** 2) / np.sum(self[-1] ** 2)
        else:
            return np.inf


class ListBackend(Backend):  # pylint: disable=too-many-ancestors
    """A Backend that use python list for storage"""

    def __init__(self, init=None):
        """Initialise the backend.

        init: initial value
        """
        if init is None:
            self._storage = []
        else:
            self._storage = [np.asarray(init)]

    def append(self, value):
        """Append a value at the end.

        Only the last value is stored. Therefor, the method updated internal
        variables for sum, ...

        """
        self._storage.append(np.asarray(value))

    @property
    def last(self):
        return self._storage[-1]

    def sum(self, sum_start=0):
        return sum(self._storage[sum_start:])

    def mean(self, sum_start=0):
        return sum(self._storage[sum_start:]) / (len(self) - sum_start)

    def var(self, sum_start=0):
        return (sum(val ** 2 for val in self[sum_start:]) - self.mean() ** 2) / (
            len(self) - sum_start
        )

    def std(self, sum_start=None):
        return np.sqrt(self.var(sum_start))

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, key):
        return self._storage[key]


class NPYBackend(Backend):  # pylint: disable=too-many-ancestors
    """A Backend that use numpy array for storage"""

    def __init__(self, maxitem: int, init=None):
        """Initialise the backend.

        maxitem: maximum number of value storrable in the backend

        init: initial value
        """
        self.maxitem = maxitem
        self._storage = np.empty((maxitem,) + np.asarray(init).shape)
        self.length = 0
        if init is not None:
            self._storage[0] = init
            self.length = 1

    def append(self, value):
        self._storage[self.length] = value
        self.length += 1

    @property
    def last(self):
        return self._storage[self.length - 1]

    def sum(self, sum_start=0):
        return np.sum(self._storage[sum_start : self.length], axis=0)

    def mean(self, sum_start=0):
        return np.mean(self._storage[sum_start : self.length], axis=0)

    def var(self, sum_start=0):
        return np.var(self._storage[sum_start : self.length], axis=0)

    def std(self, sum_start=0):
        return np.sqrt(self.var(sum_start))

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        return self._storage[key]


class H5PYBackend(NPYBackend):  # pylint: disable=too-many-ancestors
    """A Backend that use a hdf5 file for storage"""

    def __init__(
        self,
        maxitem: int,
        init=None,
        filepath: str = None,
    ):
        """Initialise the backend.

        sum_start: the index at which the sum of values must start

        init: initial value

        """
        if filepath is None:
            self.filepath = tempfile.NamedTemporaryFile().name

        self._h5file = h5py.File(self.filepath)
        self.maxitem = maxitem
        self._storage = self._h5file.create_dataset(
            "trace", data=init[np.newaxis], maxshape=(maxitem,) + init.shape
        )
        if init is not None:
            self._storage[0] = np.asarray(init)
            self.length = 1


class HollowBackend(Backend):  # pylint: disable=too-many-ancestors
    """An Hollow backend that does not store all values

    The backend does not store all the appended values but only the last one. It
    keep also internally the accumulated sum and squared accumulated sum in
    order to reply to `sum`, `mean`, `std` and `delta` methods.

    The values are stored in memory as `numpy.array`.

    Since the backend implements a Sequence, this is the last value that is
    returned with the `backend[idx]` call.

    """

    def __init__(self, sum_start=0, init=None):
        """Initialise the backend.

        sum_start: the index at which the sum of values must start

        init: initial value
        """
        self.sum_start = sum_start

        if init is None:
            self._val = None
            self.length = 0
        else:
            self._val = np.asarray(init)
            self.length = 1

        if init is not None and sum_start == 0:
            self._sum = self._val
            self._sum2 = self._val ** 2
            self.sum_count = 1
        else:
            self._sum = np.zeros_like(init)
            self._sum2 = np.zeros_like(init)
            self.sum_count = 0

        self._delta = np.inf

    def append(self, value):
        """Append a value at the end.

        Only the last value is stored. Therefor, the method updated internal
        variables for sum, ...

        """
        self._val = np.asarray(value)

        if self.length >= self.sum_start:
            self.sum_count += 1
            self._sum = self._sum + self._val
            self._sum2 = self._sum2 + self._val ** 2

        if self._val is not None and (denom := np.sum(self._val ** 2)) != 0:
            self._delta = np.sum((self._val - value) ** 2) / denom

        self.length += 1

    @property
    def last(self):
        return self._val

    def sum(self):
        return self._sum

    def mean(self):
        return self._sum / self.sum_count

    def var(self):
        return self._sum2 / self.sum_count - self.mean() ** 2

    def std(self):
        return np.sqrt(self.var())

    @property
    def delta(self):
        return self._delta

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if key not in range(len(self)):
            raise IndexError
        else:
            return self._val


#%% Feedbacks
class Feedback:
    def __init__(self, name):
        self.name = name

    def init(self):
        pass

    def show(self, iteration, min_iter, max_iter):
        print("Iter {} [{}] / {}".format(iteration, min_iter, max_iter))

    def close(self):
        pass


class Notification(Feedback):
    def __init__(self, name):
        if not notify2.initted:
            notify2.init("Sampling")

        self.name = name
        self.notif = notify2.Notification(self.name, "Iteration ? / ?")

    @staticmethod
    def filled_bar_str(iteration, min_iter, max_iter):
        return "{}{}".format(
            round(iteration / max_iter * 10) * "■",
            (10 - round(iteration / max_iter * 10)) * "□",
        )

    def show(self, iteration, min_iter, max_iter):
        self.notif.update(
            self.name,
            "Iteration {} [{}] / {} {}".format(
                iteration,
                min_iter,
                max_iter,
                Notification.filled_bar_str(iteration, min_iter, max_iter),
            ),
        )
        self.notif.show()

    def close(self):
        self.notif.close()


class Bar(Feedback):
    def __init__(self, name=""):

        self.name = name

    def show(self, iteration, min_iter, max_iter):
        if hasattr(self, "bar"):
            self.bar.update(n=1)
        else:
            self.bar = tqdm.tqdm(desc=self.name, total=max_iter, dynamic_ncols=True)

    def close(self):
        self.bar.close()


class Figure(Feedback):
    def __init__(self, fig, name="", plotters=None, traces=None, stochastic=True):
        super().__init__(name)
        self.fig = fig
        self.plotters_list = [] if plotters is None else plotters
        self.traces_list = [] if traces is None else traces
        plt.show(block=False)
        self.stochastic = stochastic
        self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        for axe in self.fig.get_axes():
            axe.cla()
        # self.init()

    def init(self):
        pass
        # for axe in self.fig.get_axes():
        #     axe.cla()

    def show(self, iteration, min_iter, max_iter):
        for plotter, trace in zip(self.plotters_list, self.traces_list):
            plotter.update(trace)
            if self.stochastic:
                plotter.update_stoch(trace)

        self.fig.suptitle(
            "{} : [{}] <= {} / {}".format(self.name, min_iter, iteration, max_iter)
        )

        if iteration == 1:
            self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


#%% Looper
class Looper(collections.abc.Iterator):
    def __init__(
        self,
        max_iter,
        min_iter,
        feedbacks=None,
        stop_fct=None,
        callback=None,
        speedrun=True,
    ):
        self.max_iter = max_iter
        self.min_iter = min_iter
        if self.min_iter > self.max_iter:
            warnings.warn(
                "Max iteration ({0}) is lower than min iteration ({1}). "
                "Max is set to min".format(self.max_iter, self.min_iter)
            )
            self.max_iter = self.min_iter
        self.nit = 0
        self.status = 2
        self.message = "Running"
        self.succes = False
        self.timestamp = []
        self.feedbacks_duration = []

        self.speedrun = speedrun
        self.callback = callback

        if stop_fct is not None:
            self.stop = stop_fct
        else:
            self.stop = lambda: False

        if isinstance(feedbacks, collections.Iterable):
            self.feedbacks = feedbacks
        elif feedbacks is None:
            self.feedbacks = []
        else:
            self.feedbacks = [feedbacks]

    @property
    def time(self):
        """Return effective time of controlled iteration"""
        arr = np.array(self.timestamp) - self.timestamp[0]
        return arr - np.cumsum(self.feedbacks_duration)

    @property
    def mean_time(self):
        """Return effective mean time of controlled iteration"""
        return np.mean(np.diff(self.time))

    def __iter__(self):
        self.nit = 0
        self.status = 2
        self.message = "Running"
        self.succes = False
        self.timestamp = [time.time()]
        self.feedbacks_duration = [0]
        for feedback in self.feedbacks:
            feedback.init()
        return self

    def __next__(self):
        if (self.nit >= self.min_iter) and self.stop():
            for feedback in self.feedbacks:
                feedback.show(self.nit, self.min_iter, self.max_iter)
                feedback.close()
            self.status = 0
            self.message = "Condition reached"
            self.succes = True
            raise StopIteration()
        elif self.nit < self.max_iter:
            if not self.speedrun:
                tic = time.time()
                for feedback in self.feedbacks:
                    feedback.show(self.nit, self.min_iter, self.max_iter)
                self.feedbacks_duration.append(time.time() - tic)
            else:
                self.feedbacks_duration.append(0)

            self.timestamp.append(time.time())
            self.nit += 1
            return self.nit
        else:
            for feedback in self.feedbacks:
                feedback.close()
            self.status = 1
            self.message = "Maximum iteration reached"
            self.succes = False
            raise StopIteration()

    def __repr__(self):
        fb = np.mean(self.feedbacks_duration)
        return (
            "Succes : {}; {}\n".format(self.succes, self.message)
            + "[{}] <= {} / {}\n".format(self.min_iter, self.nit, self.max_iter)
            + "Total time {:.2g} / mean time {:.2g} / FB time {:.2g} ({:.1f}%)".format(
                self.time[-1], self.mean_time, fb, 100 * fb * (fb + self.mean_time)
            )
        )


class IterativeAlg:
    def __init__(
        self,
        max_iter,
        min_iter,
        stochastic,
        threshold=1e-6,
        speedrun=True,
        feedbacks=None,
        trace=HollowTrace,
        name="",
    ):

        if isinstance(feedbacks, collections.Iterable):
            for fb in feedbacks:
                fb.name = name
                fb.stochastic = stochastic
                if isinstance(fb, Figure):
                    self._figure = fb
                else:
                    self._figure = None
        elif isinstance(feedbacks, Figure):
            feedbacks.name = name
            feedbacks.stochastic = stochastic
            if isinstance(feedbacks, Figure):
                self._figure = feedbacks
            else:
                self._figure = None

        self.looper = Looper(max_iter, min_iter, speedrun=speedrun, feedbacks=feedbacks)

        self.stochastic = stochastic
        self.threshold = threshold
        self.name = name
        self._trace = trace

    @property
    def max_iter(self):
        return self.looper.max_iter

    @property
    def min_iter(self):
        return self.looper.min_iter

    def stop(self, obj):
        if self.stochastic:
            if obj.mean_delta < self.threshold:
                return True
            else:
                return False
        else:
            if obj.delta < self.threshold:
                return True
            else:
                return False

    def fig_register_trace(self, *args):
        if hasattr(self, "_figure"):
            axes = self._figure.fig.get_axes()
            self._figure.traces_list = args
            for axe, trace in zip(axes, args):
                if trace.ndim == 0:
                    self._figure.plotters_list.append(
                        MplScalarTracePlot(axe, trace.name)
                    )
                elif trace.ndim == 1:
                    self._figure.plotters_list.append(Mpl1DTracePlot(axe, trace.name))
                elif trace.ndim == 2:
                    self._figure.plotters_list.append(Mpl2DTracePlot(axe, trace.name))
                else:
                    print("Trace DataType not supported for plotting")

    def watch_for_stop(self, trace):
        self.looper.stop = functools.partial(self.stop, trace)


class IterRes:
    def __init__(self, looper, **kwargs):
        self.looper = looper
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.status = looper.status
        self.message = looper.message
        self.succes = looper.succes
        self.nit = self.looper.nit
        self.max_iter = looper.max_iter
        self.min_iter = looper.min_iter
        self.time = list(looper.time)
        self.mean_time = looper.mean_time
        self.fb_time = np.mean(looper.feedbacks_duration)
        self.fun = 0
        self.jac = 0
        self.hess = 0
        self.hess_inv = 0
        self.nfev = 0
        self.njev = 0
        self.maxcv = 0

    @property
    def x(self):
        return self.minimizer

    def __repr__(self):
        return self.looper.__repr__()
