Welcome to the TorchTNT documentation!
===========================================

TNT is a library for PyTorch training tools and utilities. It has two main components, which are the top-level modules of the repo:

1. :mod:`torchtnt.framework`: contains a lightweight training framework to simplify maintaining training, evaluation, and prediction loops.
2. :mod:`torchtnt.utils`: contains a grab-bag of various independent, training-related utilities, including data related abstractions and wrappers around different publishers to simplify logging metrics.

Installation
--------------

TNT can be installed with pip. To do so, run:

.. code-block:: shell

   pip install torchtnt

If you run into issues, make sure that Pytorch is installed first.

You can also install the latest version from master. Just run:

.. code-block:: shell

   pip install git+https://github.com/pytorch/tnt.git@master

To update to the latest version from master:

.. code-block:: shell

   pip install --upgrade git+https://github.com/pytorch/tnt.git@master


Documentation
---------------
.. toctree::
      :maxdepth: 1
      :caption: Overview
      :glob:

      overview

.. fbcode::

   .. toctree::
      :maxdepth: 2
      :caption: Getting Started (Meta)
      :glob:

      meta/getting_started
      meta/migrating
      meta/migrating_example
      meta/tss_to_dcp

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples

.. fbcode::

   .. toctree::
      :maxdepth: 2
      :caption: Examples (Meta)
      :glob:

      meta/examples

.. fbcode::

   .. toctree::
      :maxdepth: 2
      :caption: Debugging FAQ (Meta)
      :glob:

      meta/checkpointing_FAQ
      meta/mem_debug

.. toctree::
   :maxdepth: 1
   :caption: Core Concepts

   distributed
   checkpointing

.. toctree::
   :maxdepth: 1
   :caption: Framework

   framework/unit
   framework/auto_unit
   framework/train
   framework/eval
   framework/predict
   framework/fit
   framework/state
   framework/callbacks

.. toctree::
   :maxdepth: 2
   :caption: Utils

   utils/utils

.. fbcode::

   .. toctree::
      :maxdepth: 2
      :caption: Utils (Meta)
      :glob:

      meta/utils
