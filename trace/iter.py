#!/usr/bin/env python3

from collections.abc import Iterator, Iterable

import tqdm

import numpy as np
import matplotlib.pyplot as plt
import warnings

from . import plot

try:
    import notify2
except ImportError:
    pass
else:
    if not notify2.initted:
        notify2.init("Trace active")


#%% Feedbacks
class Feedback:
    def __init__(self, min_iter=0, max_iter=1, name=""):
        self.name = name
        self.min_iter = min_iter
        self.max_iter = max_iter

    def init(self):
        pass

    def show(self, iteration, min_iter, max_iter):
        print("Iter {} [{}] / {}".format(iteration, min_iter, max_iter))

    def close(self):
        pass


class Notification(Feedback):
    def __init__(self, min_iter=0, max_iter=1, name=""):
        super().__init__(min_iter, max_iter, name)

        if not notify2.initted:
            notify2.init("Starting")
        self.notif = notify2.Notification(self.name, "Iteration ? / ?")

    @staticmethod
    def filled_bar_str(iteration):
        return "{}{}".format(
            round(iteration / self.max_iter * 10) * "■",
            (10 - round(iteration / self.max_iter * 10)) * "□",
        )

    def show(self, iteration):
        self.notif.update(
            self.name,
            "Iteration {} [{}] / {} {}".format(
                iteration,
                self.min_iter,
                self.max_iter,
                Notification.filled_bar_str(iteration, self.min_iter, self.max_iter),
            ),
        )
        self.notif.show()

    def close(self):
        self.notif.close()


class Bar(Feedback):
    def __init__(self, min_iter=0, max_iter=1, name=""):
        super().__init__(min_iter, max_iter, name)

        self.last_iter = 0
        self.bar = tqdm.tqdm(desc=self.name, total=self.max_iter, dynamic_ncols=True)

    def init(self):
        self.close()
        self.bar = tqdm.tqdm(desc=self.name, total=self.max_iter, dynamic_ncols=True)

    def show(self, iteration):
        self.bar.update(n=iteration - self.last_iter)
        self.last_iter = iteration

    def close(self):
        self.bar.close()


class Figure(Feedback):
    def __init__(self, fig, min_iter=0, max_iter=1, name=""):
        super().__init__(min_iter, max_iter, name)

        self.fig = fig
        plt.show(block=False)
        self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        for axe in self.fig.get_axes():
            axe.cla()

    def init(self):
        pass
        # for axe in self.fig.get_axes():
        #     axe.cla()

    def show(self, iteration):
        self.fig.suptitle(
            f"{self.name} : [{self.min_iter}] <= {iteration} / {self.max_iter}"
        )

        if iteration == 1:
            self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


#%% Looper
class Looper(Iterator):
    def __init__(
        self,
        max_iter,
        min_iter,
        feedbacks=None,
        stop_fct=None,
        speedrun=True,
    ):
        self.max_iter = max_iter
        self.min_iter = min_iter
        if self.min_iter > self.max_iter:
            warnings.warn(
                "Max iteration ({self.max_iter}) is lower than min iteration ({self.min_iter}). "
                "Max is set to min"
            )
            self.max_iter = self.min_iter
        self.nit = 0
        self.status = 2
        self.message = "Running"
        self.succes = False
        self.timestamp = []
        self.feedbacks_duration = []

        self.speedrun = speedrun

        if stop_fct is not None:
            self.stop = stop_fct
        else:
            self.stop = lambda: False

        if isinstance(feedbacks, Iterable):
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
                feedback.show(self.nit)
                feedback.close()
            self.status = 0
            self.message = "Condition reached"
            self.succes = True
            raise StopIteration()
        elif self.nit < self.max_iter:
            if not self.speedrun:
                tic = time.time()
                for feedback in self.feedbacks:
                    feedback.show(self.nit)
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
            "Success : {}; {}\n".format(self.succes, self.message)
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

        if isinstance(feedbacks, Iterable):
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

    def plot_trace(self, fig, *args):
        self.plotters = []
        for axe, trace in zip(fig.get_axes(), args):
            if trace.ndim == 0:
                self.plotters.append(plot.MplScalarTracePlot(axe, trace.name))
            elif trace.ndim == 1:
                self.plotters.append(plot.Mpl1DTracePlot(axe, trace.name))
            elif trace.ndim == 2:
                self.plotters.append(plot.Mpl2DTracePlot(axe, trace.name))
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
