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

import time
import abc
import functools as ft

import matplotlib.cm as cm
import numpy as np

from .trace import StochTrace

__author__ = "François Orieux"
__copyright__ = "2018-2020 F. Orieux <orieux@l2s.centralesupelec.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__maintainer__ = "François Orieux"
__email__ = "orieux@l2s.centralesupelec.fr"
__status__ = "early alpha"
__url__ = ""
__keywords__ = "tools, algorithmes"


def timed(obj, func):
    @ft.wraps(func)
    def with_time():
        tic = time.time()
        func()
        obj.duration.append(time.time() - tic)

    return with_time


#%% Traces plotter
class TracePlot(abc.ABC):
    def __init__(self, axe):
        self.axe = axe
        self.fig = self.axe.figure

        axe.set_title("{}".format(self.name))

        self.duration = []
        self.update = timed(
            self, self.update
        )  # Add timing to all update methods of subclass

    def name(self):
        if hasattr(self, "_trace"):
            return self._trace.name
        else:
            return ""

    @property
    def trace(self):
        return self.get_trace()

    def get_trace(self):
        if hasattr(self, "_trace"):
            return self._trace
        else:
            return None

    def set_trace(self, val):
        self._trace = val
        self._trace.register(self)

    @abstractmethod
    def update(self):
        raise NotImplemented


class MplScalarTracePlot(TracePlot):
    def __init__(self, axe):
        super().__init__(axe)

        axe.set_xlabel("Iteration")

    def set_trace(self, trace):
        super().set_trace(trace)

        (self.line,) = self.axe.plot(self.trace)
        if isinstance(self.trace, StochTrace):
            self.burnin_line = self.axe.axvline(self.trace.burnin, ls="--", alpha=0.2)
            (self.mean_line,) = self.axe.plot(trace.cum_mean(), color="red", lw=2)
            self.noise_fill = self.axe.fill_between(
                np.arange(trace.burnin, len(trace)),
                trace.cum_mean() - trace.cum_std(),
                trace.cum_mean() + trace.cum_std(),
                facecolor="blue",
                alpha=0.2,
            )

    def update(self):
        self.line.set_ydata(self.trace)
        self.line.set_xdata(np.arange(len(self.trace)))
        self.axe.set_title("{}: {:.2g}".format(self.name, self.trace.last))
        self.axe.set_xlim(0, max(len(self.trace), self.trace.burnin + 10))
        self.axe.set_ylim(0.9 * min(self.trace), 1.1 * max(self.trace))

        if isinstance(self.trace, StochTrace) and self.trace.burned:
            self.noise_fill.remove()

            self.mean_line.set_ydata(self.trace.cum_mean())
            self.mean_line.set_xdata(
                np.arange(len(self.trace.cum_mean())) + self.trace.burnin
            )
            self.noise_fill = self.axe.fill_between(
                np.arange(self.trace.burnin, len(self.trace)),
                self.trace.cum_mean() - self.trace.cum_std(),
                self.trace.cum_mean() + self.trace.cum_std(),
                facecolor="blue",
                alpha=0.2,
            )

            self.axe.set_title(
                "{}: {:.2g} / {:.2g} +- {:.2g}".format(
                    self.name, self.trace.last, self.trace.mean(), self.trace.std()
                )
            )
        else:
            self.axe.set_title("{}: {:.2g}".format(self.name, self.trace.last))


class Mpl1DTracePlot(TracePlot):
    def __init__(self, axe):
        super().__init__(axe)
        self.axe.set_xlabel("'n'")

    def set_trace(self, trace):
        super().set_trace(trace)
        (self.line,) = self.axe.plot(trace.last)
        (self.mean_line,) = self.axe.plot(trace.mean(), color="red", alpha=1)
        self.uq_fill = self.axe.fill_between(
            np.arange(len(trace.last)),
            trace.mean() - trace.std(),
            trace.mean() + trace.std(),
            facecolor="blue",
            alpha=0.2,
        )

    def update(self):
        self.line.set_ydata(self.trace.last)
        self.line.set_xdata(np.arange(len(self.trace.last)))

        if isinstance(self.trace, StochTrace) and self.trace.burned:
            self.uq_fill.remove()
            self.mean_line.set_ydata(self.trace.mean())
            self.uq_fill = self.axe.fill_between(
                np.arange(len(self.trace.last)),
                self.trace.mean() - self.trace.std(),
                self.trace.mean() + self.trace.std(),
                facecolor="blue",
                alpha=0.2,
            )
            self.axe.set_ylim(
                0.9 * min(self.trace.mean()), 1.1 * max(self.trace.mean())
            )
            self.axe.set_title(
                "{} / Δμ: {:.1g} / Δσ: {:.1g}".format(
                    self.name, self.trace.mean_delta, self.trace.std_delta
                )
            )
        else:
            self.axe.set_ylim(1.1 * min(self.trace.last), 1.1 * max(self.trace.last))
            self.axe.set_title("{} / Δ: {:.1g}".format(self.name, self.trace.delta))


class Mpl2DTracePlot(TracePlot):
    def __init__(self, axe, preproc=abs):
        super().__init__(axe)
        self.axe.set_axis_off()
        self.preproc = preproc

    def set_trace(self, trace):
        super().set_trace(trace)
        self.img = self.axe.imshow(abs(trace.last), cmap=cm.gray)

    def update(self):
        if isinstance(self.trace, StochTrace) and self.trace.burned:
            val = self.preproc(self.trace.mean())
            self.axe.set_title(
                "{} / Δμ: {:.1g} / Δσ: {:.1g}".format(
                    self.name, self.trace.mean_delta, self.trace.std_delta
                )
            )
        else:
            val = self.preproc(self.trace.last)
            self.axe.set_title("{} / Δ: {:.1g}".format(self.name, self.trace.delta))

        self.img.set_data(val)
        self.axe.set_aspect("auto")
        self.img.autoscale()


class Mpl2DStdTracePlot(TracePlot):
    def __init__(self, axe, trace, preproc=abs):
        super().__init__(axe, trace)
        self.axe.set_axis_off()
        self.preproc = preproc
        self.axe.set_title(f"Σ {self.name}")

    def set_trace(self, trace):
        super().set_trace(trace)
        self.img = self.axe.imshow(np.zeros_like(trace.last), cmap=cm.gray)

    def update(self):
        if isinstance(self.trace, StochTrace) and self.trace.burned:
            val = self.preproc(self.trace.std())
            self.img.set_data(val)
            self.axe.set_aspect("auto")
            self.img.autoscale()
