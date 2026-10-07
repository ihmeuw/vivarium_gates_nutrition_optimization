# Check if we're running in Jenkins
ifdef JENKINS_URL
# 	Files are already in workspace from shared library
	MAKE_INCLUDES := .
else
# 	For local dev, use the installed vivarium.build_utils package if it exists
# 	First, check if we can import vivarium.build_utils and assign 'yes' or 'no'.
# 	We do this by importing the package in python and redirecting stderr to the null device.
# 	If the import is successful (&&), it will print 'yes', otherwise (||) it will print 'no'.
	VIVARIUM_BUILD_UTILS_AVAILABLE := $(shell python -c "import vivarium.build_utils" 2>/dev/null && echo "yes" || echo "no")
# 	If vivarium.build_utils is available, get the makefiles path or else set it to empty
	ifeq ($(VIVARIUM_BUILD_UTILS_AVAILABLE),yes)
		MAKE_INCLUDES := $(shell python -c "from vivarium.build_utils.resources import get_makefiles_path; print(get_makefiles_path())")
	else
		MAKE_INCLUDES :=
	endif
endif

# Set the package name as the last part of this file's parent directory path
PACKAGE_NAME = $(notdir $(CURDIR))

# Helper function for validating enum arguments
validate_arg = $(if $(filter-out $(2),$(1)),$(error Error: '$(3)' must be one of: $(2), got '$(1)'))

# Environment selector for conda: a prefix when `path` is given, otherwise a name.
CONDA_ENV_FLAG = $(if $(path),-p $(path),-n $(name))

# Extras group that `type` selects from base.mk's `install` target.
ENV_REQS_FOR_TYPE = $(if $(filter artifact,$(type)),data,dev)

# Macro for validating make target arguments
# Usage: $(call validate_make_args,target_name,allowed_args)
# Example: $(call validate_make_args,build-env,type name path)
define validate_make_args
	@allowed="$(2)"; \
	for arg in $(filter-out $(1),$(MAKECMDGOALS)) $(MAKEFLAGS); do \
		case $$arg in \
			*=*) \
				arg_name=$${arg%%=*}; \
				if ! echo " $$allowed " | grep -q " $$arg_name "; then \
					allowed_list=$$(echo $$allowed | sed 's/ /, /g'); \
					echo "Error: Invalid argument '$$arg_name'. Allowed arguments are: $$allowed_list" >&2; \
					exit 1; \
				fi \
				;; \
		esac; \
	done
endef

ifneq ($(MAKE_INCLUDES),) # not empty
# Include makefiles from vivarium_build_utils
include $(MAKE_INCLUDES)/base.mk
include $(MAKE_INCLUDES)/test.mk
else # empty
# Use this help message (since the vivarium_build_utils version is not available)
.PHONY: help
help:
	@echo
	@echo "For Make's standard help, run 'make --help'."
	@echo
	@echo "Most of our Makefile targets are provided by the vivarium_build_utils"
	@echo "package. To access them, you need to create a development environment first."
	@echo
	@echo "================================================================================"
	@echo "build-env: Create a full conda environment from scratch"
	@echo "================================================================================"
	@echo
	@echo "This target creates a new conda environment and installs all required"
	@echo "packages for development or artifact generation, depending on the 'type' argument."
	@echo
	@echo "USAGE:"
	@echo "  make build-env [type=<environment type>] [name=<environment name>] [path=<environment path>] [py=<python version>] [include_timestamp=<yes|no>] [lfs=<yes|no>] [force=<yes|no>] [keep_env=<yes|no>]"
	@echo
	@echo "ARGUMENTS:"
	@echo "  type [optional]"
	@echo "      Type of conda environment. Either 'simulation' (default) or 'artifact'"
	@echo "  name [optional]"
	@echo "      Name of the conda environment to create (defaults to <PACKAGE_NAME>_<TYPE>)"
	@echo "  path [optional]"
	@echo "      Absolute path where the environment should be created (overrides name for location)"
	@echo "  include_timestamp [optional]"
	@echo "      Whether to append a timestamp to the environment name. Either 'yes' or 'no' (default)"
	@echo "  lfs [optional]"
	@echo "      Whether to install git-lfs in the environment. Either 'yes' or 'no' (default)"
	@echo "  py [optional]"
	@echo "      Python version (defaults to latest supported)"
	@echo "  force [optional]"
	@echo "      Whether to remove and recreate an existing environment. Either 'yes' or 'no' (default)"
	@echo "  keep_env [optional]"
	@echo "      Whether to keep a failed build's partial environment for inspection."
	@echo "      Either 'yes' or 'no' (default, i.e. tear it down)."
	@echo
	@echo "After creating the environment:"
	@echo "  1. Activate it: 'conda activate <environment_name>'"
	@echo "  2. Run 'make help' again to see all newly available targets"
	@echo
endif

.PHONY: build-env
build-env: # Create a new environment with installed packages
#	Validate arguments - exit if unsupported arguments are passed
	$(call validate_make_args,build-env,type name path lfs py include_timestamp force keep_env)
	
#   Handle arguments and set defaults
#   type
	@$(eval type ?= simulation)
	@$(call validate_arg,$(type),simulation artifact,type)
#	name
	@$(eval name ?= $(PACKAGE_NAME)_$(type))
#	timestamp
	@$(eval include_timestamp ?= no)
	@$(call validate_arg,$(include_timestamp),yes no,include_timestamp)
	@$(if $(filter yes,$(include_timestamp)),$(eval override name := $(name)_$(shell date +%Y%m%d_%H%M%S)),)
#	path (optional - if set, use -p for conda create instead of -n)
	@$(eval path ?=)
#	lfs
	@$(eval lfs ?= no)
	@$(call validate_arg,$(lfs),yes no,lfs)
#	force
	@$(eval force ?= no)
	@$(call validate_arg,$(force),yes no,force)
#	keep_env
	@$(eval keep_env ?= no)
	@$(call validate_arg,$(keep_env),yes no,keep_env)
#	python version
	@$(eval py ?= $(shell cat python_versions.json | tr -d '[]" ' | tr ',' '\n' | sort -t. -k1,1n -k2,2n | tail -1))

#	Check if environment already exists and handle based on force flag
	@if conda env list | grep -qE "$(if $(path),^$(path),^$(name))\s"; then \
		if [ "$(force)" = "yes" ]; then \
			echo "Removing existing environment..."; \
			conda remove $(CONDA_ENV_FLAG) --all --yes; \
		else \
			echo "Error: Environment already exists at $(if $(path),$(path),$(name))" >&2; \
			echo "Use 'force=yes' to remove and recreate it, or specify a different location with 'name=<name>' or 'path=<path>'" >&2; \
			exit 1; \
		fi \
	fi

#	The force check above guarantees we only tear down an env this invocation created.
	@if ! $(MAKE) --no-print-directory _build-env \
			name=$(name) path=$(path) py=$(py) type=$(type) lfs=$(lfs); then \
		echo >&2; \
		if [ "$(keep_env)" = "yes" ]; then \
			echo "Error: failed to build environment $(if $(path),$(path),$(name)). Keeping it (keep_env=yes)." >&2; \
		else \
			echo "Error: failed to build environment $(if $(path),$(path),$(name)). Removing it; pass keep_env=yes to keep it for inspection." >&2; \
			conda env remove $(CONDA_ENV_FLAG) --yes; \
		fi; \
		exit 1; \
	fi

	@echo
	@echo "Finished building environment"
	@$(if $(path),echo "  path: $(path)",echo "  name: $(name)")
	@echo "  type: $(type)"
	@echo "  git-lfs installed: $(lfs)"
	@echo "  python version: $(py)"
	@echo "  forced rebuild: $(force)"
	@echo
	@echo "After creating the environment:"
	@$(if $(path),echo "  1. Activate it: 'conda activate $(path)'",echo "  1. Activate it: 'conda activate $(name)'")
	@echo "  2. Run 'make help' again to see all newly available targets"
	@echo

# Private: the build steps live here so `build-env` can tear down the env if any fails.
.PHONY: _build-env
_build-env:
	conda create $(CONDA_ENV_FLAG) python=$(py) --yes
# 	--no-capture-output streams output, so a failure isn't hidden behind a long silence.
	conda run --no-capture-output $(CONDA_ENV_FLAG) pip install "vivarium_build_utils>=4.0.0,<5.0.0"
	conda run --no-capture-output $(CONDA_ENV_FLAG) make install ENV_REQS=$(ENV_REQS_FOR_TYPE)
	@if [ "$(type)" = "simulation" ]; then \
		conda install $(CONDA_ENV_FLAG) redis -c anaconda -y; \
	fi
# 	`set -e` is load-bearing: a multi-command line reports only its last command's status.
	@if [ "$(lfs)" = "yes" ]; then \
		set -e; \
		conda run --no-capture-output $(CONDA_ENV_FLAG) conda install -c conda-forge git-lfs --yes; \
		conda run --no-capture-output $(CONDA_ENV_FLAG) git lfs install; \
	fi
# 	A fresh env's only editable install is this checkout, so an empty list means it never installed.
	@conda run --no-capture-output $(CONDA_ENV_FLAG) pip list --editable --format=freeze | grep -q . \
		|| { echo "Error: nothing was installed into $(if $(path),$(path),$(name)) from $(CURDIR)." >&2; exit 1; }

