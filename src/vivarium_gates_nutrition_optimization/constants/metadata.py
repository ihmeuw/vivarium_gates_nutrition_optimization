from typing import NamedTuple

import pandas as pd

####################
# Project metadata #
####################

PROJECT_NAME = "vivarium_gates_nutrition_optimization"
CLUSTER_PROJECT = "proj_simscience_prod"

CLUSTER_QUEUE = "all.q"
MAKE_ARTIFACT_MEM = "10G"
MAKE_ARTIFACT_CPU = "1"
MAKE_ARTIFACT_RUNTIME = "3:00:00"
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = ["Ethiopia", "Nigeria", "Pakistan"]

ARTIFACT_INDEX_COLUMNS = [
    "sex",
    "age_start",
    "age_end",
    "year_start",
    "year_end",
]

DRAW_COUNT = 500
ARTIFACT_COLUMNS = pd.Index([f"draw_{i}" for i in range(DRAW_COUNT)])


class __Scenarios(NamedTuple):
    zero_coverage: str
    baseline: str
    mms: str
    universal_bep: str
    targeted_bep_ifa: str
    targeted_bep_mms: str


SCENARIOS = __Scenarios(*__Scenarios._fields)
