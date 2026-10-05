**11.1 - 10/05/26**

 - Record the monorepo migration (#152, #154, #155, #156, #157, #159): move the build system from setup.py to pyproject.toml, depend on vivarium-engine/vivarium-public-health/vivarium-gbd-mapping and vivarium-build-utils 4.x instead of the single-package vivarium 4.x, vivarium_public_health 5.x and gbd_mapping, and fix the model for GBD 2023 V&V
 - Replace the Makefile with the shared vivarium model version: the distribution name is read from pyproject.toml, build-env gains path= and force= arguments and refuses to clobber an existing environment, and build-shared-env and print-dist-name targets are added
 - Replace environment.sh: it now derives the environment name from pyproject.toml, builds through make, supports -s for a venv overlay on the Jenkins shared environment, and must be sourced
 - Remove requirements.txt, artifact_requirements.txt, .flake8, and pytype.cfg; environments install from the pyproject extras only
 - Add vivarium_gbd_access>=6.0.0,<7.0.0 to the data extra and include the test extra in it
 - Add a [tool.uv] override-dependencies block pinning pandas<3, numpy<2, and sqlalchemy 2.x
 - Rewrite the README Installation section for the local and shared environment workflows

**11.0 - 3/26/24**

 - Update to 2021 data for Ethiopia, bug-fix changes to pregnancy observation, changes to stratification (anemia by pregnancy state and pregnancy transitions by pregnancy outcomes), temporary removal of the IFA scenario, and removal of background morbidity.

**1.0.3 - 3/12/24**

 - Update Mortality observer to incorporate update in VPH Mortality Observer

**1.0.2 - 03/11/24**

 - Remove universal BEP scenario and targeted BEP/no MMS scenario and include universal IFA scenario and targeted BEP/IFA scenario

**1.0.1 - 10/02/23**

 - Refactor to use vivarium Components

**1.0.0 - 09/18/23**

 - Release to 1.0.0 with Wave 1 production runs
