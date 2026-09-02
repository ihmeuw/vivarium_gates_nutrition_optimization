=====================================
vivarium_gates_nutrition_optimization
=====================================

.. contents::
   :depth: 1

Installation
------------

You will need ``conda`` installed in order to install the requirements from this repository. 
You should follow these instructions for
your operating system:

- `conda <https://docs.conda.io/en/latest/miniconda.html>`_   

Once you have this installed, you should open up your normal shell
(if you're on linux or OSX) or the ``git bash`` shell if you're on windows.
Within this shell, navigate to the simulation directory. 
The simulation directory is where this README file is located and will be titled 
something like ihmeuw-vivarium_gates_nutrition_optimization-{hash}
You will then then make an environment and install
all necessary requirements as follows::

   cd /FILE/PATH/TO/SIMULATION/DIRECTORY/
   conda create --name vivarium_gates_nutrition_optimization --file vivarium_gates_nutrition_optimization_child_lock.txt
   conda activate vivarium_gates_nutrition_optimization
   pip install -r vivarium_gates_nutrition_optimization_lock_pip.txt
   pip install -e . 

Note the ``-e`` flag that follows pip install. This will install the python
package in-place, which is important for making the model specifications later.

Vivarium uses the Hierarchical Data Format (HDF) as the backing storage
for the data artifacts that supply data to the simulation. You may not have
the needed libraries on your system to interact with these files, and this is
not something that can be specified and installed with the rest of the package's
dependencies via ``pip``. If you encounter HDF5-related errors, you should
install hdf tooling from within your environment like so::

  (vivarium_gates_nutrition_optimization) :~$ conda install hdf5

The ``(vivarium_gates_nutrition_optimization)`` that precedes your shell prompt will probably show
up by default, though it may not.  It's just a visual reminder that you
are installing and running things in an isolated programming environment
so it doesn't conflict with other source code and libraries on your
system.


Usage
-----

You'll find six directories inside the main
``src/vivarium_gates_nutrition_optimization`` package directory:

- ``artifacts``

  This directory contains all input data used to run the simulations.
  You can open these files and examine the input data using the vivarium
  artifact tools.  A tutorial can be found at https://vivarium.readthedocs.io/en/latest/tutorials/artifact.html#reading-data

- ``components``

  This directory is for Python modules containing custom components for
  the vivarium_gates_nutrition_optimization project. You should work with the
  engineering staff to help scope out what you need and get them built.

- ``data``

  If you have **small scale** external data for use in your sim, it can live here.
  This is almost certainly not the right place for data, so make sure there's not
  a better place to put it first.

- ``model_specifications``

  This directory should hold all model specifications and branch files
  associated with the project.

- ``tools``

  This directory hold Python files used to run scripts used to prepare input
  data or process outputs.

Running Simulations
-------------------

You can run your simulation from the command line. 
With your conda environment active, you can run with, e.g.::

   (vivarium_gates_nutrition_optimization) :~$ cd /FILE/PATH/TO/SIMULATION/DIRECTORY/
   (vivarium_gates_nutrition_optimization) :~$ simulate run -vvv src/vivarium_gates_nutrition_optimization/model_specifications/model_spec.yaml -o /FILE/PATH/TO/SAVE/RESULTS -i src/vivarium_gates_nutrition_optimization/artifacts/<COUNTRY_TO_RUN_IN>.hdf

The simulation will run in one location at a time, enter the country you wish to 
run the simulation for in your call. Ethiopia, Nigeria, and Paksitan are supported. 
The country name should be in lower case, for example 'ethiopia' or 'nigeria'.  

The ``-vvv`` flag will log verbosely, so you will get log messages every time
step. For more ways to run simulations, see the tutorials at
https://vivarium.readthedocs.io/en/latest/tutorials/running_a_simulation/index.html
and https://vivarium.readthedocs.io/en/latest/tutorials/exploration.html
