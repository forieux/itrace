=========================
Trace
=========================

The iTrace (``itrace``) package provides a data structure to help prototyping
data processing algorithm. The main contribution is the ``Trace`` type that
mimic a ``list`` with additionnal features.

The package is in very alpha stage.

Info
====

* Author: `François Orieux <http://pro.orieux.fr>`_
* Contact: orieux at l2s.centralesupelec.fr
* Project homepage: http://github.com/forieux/itrace
* Downloads page: https://github.com/forieux/itrace/releases

Contents
========

trace.py
    The main file that implements the ``Trace`` type.

plot.py
    Tools to plot `Trace`.

iter.py
    Tools for iterative algorithm. I'm not sure your interested about that.


Requirements
============

This package depends on `numpy <https://numpy.org/>`_, `h5py <https://www.h5py.org>`_, `matlotlib <https://matplotlib.org/>`_, and `tqdm <https://github.com/tqdm/tqdm>`_.


Installation
============

After git clonning, I recommand to use poetry ::

    poetry install

Development
===========

This package follow the Semantic Versionning convention http://semver.org/. To
get the development version you can clone the git repository available here
http://github.com/forieux/trace

The ongoing development depends on my time but is open. I try to fix bugs.

License
=======

``itrace`` is free software distributed under the WTFPL, see LICENSE.txt.
