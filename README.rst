=========================
Trace
=========================

The Trace (``trace``) package provides a data structure to help prototyping data
processing algorithm. The main contribution is the ``Trace`` type that mimic a
``list`` with additionnal features.

Info
====

* Author: François Orieux
* Contact: orieux at l2s.centralesupelec.fr
* Project homepage: http://github.com/forieux/trace
* Downloads page: https://github.com/forieux/releases

Contents
========

trace.py
    A module that implement the algorithm described in [2] for
    unsupervised myopic image deconvolution. However the myopic part
    is not actually available.

Requirements
============

This package depends on numpy

* Numpy version >= 1.4.1

Installation
============

The ``pip`` version::

    pip install edwin

If you have not ``pip``, download the archive, decompress it and to
install in your user path, run in a command line::

    python setup.py install --user

or for the system path, run as root::

    python setup.py install

Development
===========

This package follow the Semantic Versionning convention http://semver.org/. To
get the development version you can clone the git repository available here
http://github.com/forieux/trace

The ongoing development depends on my time but is open. I try to fix bugs.

License
=======

``trace`` is free software distributed under the WTFPL, see LICENSE.txt.
