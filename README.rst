===============================
vivarium_gates_nutrition_optimization
===============================

Research repository for the vivarium_gates_nutrition_optimization project.

.. contents::
   :depth: 1

Installation
------------

You will need ``conda`` to install all of this repository's requirements.
We recommend installing `Miniforge <https://github.com/conda-forge/miniforge>`_.

Once you have conda installed, you should open up your normal shell
(if you're on linux or OSX) or the ``git bash`` shell if you're on windows.

You'll then clone this repository and make the necessary environments.
The first step is to clone the repo::

  :~$ git clone https://github.com/ihmeuw/vivarium_gates_nutrition_optimization.git
  ...git will copy the repository from github and place it in your current directory...
  :~$ cd vivarium_gates_nutrition_optimization

There are two environment options: a **local conda environment** (for personal machines)
or a **shared environment on the cluster** with a lightweight venv wrapper.

To create or update an environment, use ``source environment.sh``. This will
automatically create the environment if it doesn't exist, or update it if it
is stale.

**Local conda environment** (default)::

  :~$ source environment.sh
  ...creates/activates the simulation conda environment...
  :~$ source environment.sh -t artifact
  ...creates/activates the artifact conda environment...

To deactivate a local conda environment, run ``conda deactivate``.

**Shared environment on the cluster** (recommended for cluster development)::

  :~$ source environment.sh -s
  ...creates/activates a venv overlay on the shared simulation environment...
  :~$ source environment.sh -s -t artifact
  ...creates/activates a venv overlay on the shared artifact environment...

To deactivate a shared cluster environment, run ``deactivate``.

Additional options are available; pass the ``-h`` flag to see them
(e.g. ``-f`` to force a rebuild, ``-l`` to install git lfs).

Alternatively, users can manually create conda environments as follows::

  :~$ conda create --name=vivarium_gates_nutrition_optimization_simulation python=3.11 git git-lfs
  ...conda will download python and base dependencies...
  :~$ conda activate vivarium_gates_nutrition_optimization_simulation
  (vivarium_gates_nutrition_optimization_simulation) :~$ pip install -e .[dev]
  ...pip will install vivarium and other requirements...
  (vivarium_gates_nutrition_optimization_simulation) :~$ conda deactivate
  :~$ conda create --name=vivarium_gates_nutrition_optimization_artifact python=3.11 git git-lfs
  ...conda will download python and base dependencies...
  :~$ conda activate vivarium_gates_nutrition_optimization_artifact
  (vivarium_gates_nutrition_optimization_artifact) :~$ pip install -e .[data]
  ...pip will install vivarium and other requirements...

Supported Python versions: 3.11

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

   (vivarium_gates_nutrition_optimization) :~$ simulate run -vvv /<REPO_INSTALLATION_DIRECTORY>/vivarium_gates_nutrition_optimization/src/vivarium_gates_nutrition_optimization/model_specifications/model_spec.yaml -o /FILE/PATH/TO/SAVE/RESULTS -i src/vivarium_gates_nutrition_optimization/artifacts/<COUNTRY_TO_RUN_IN>.hdf

The simulation will run in one location at a time, enter the country you wish to 
run the simulation for in your call. Currently Ethiopia, Nigeria, and Paksitan are supported. 
The country name should be in lower case, for example 'ethiopia' or 'nigeria'.  

The ``-vvv`` flag will log verbosely, so you will get log messages every time
step. For more ways to run simulations, see the tutorials at
https://vivarium.readthedocs.io/en/latest/tutorials/running_a_simulation/index.html
and https://vivarium.readthedocs.io/en/latest/tutorials/exploration.html
