from typing import NamedTuple

import numpy as np
import pandas as pd

from vivarium_gates_nutrition_optimization.constants import models


############################
# Disease Model Parameters #
############################
class _Durations(NamedTuple):
    ## Days
    FULL_TERM = 40 * 7
    POSTPARTUM = 6 * 7
    DETECTION = 6 * 7
    PARTIAL_TERM = 24 * 7


DURATIONS = _Durations()

INFANT_MALE_PERCENTAGES = {
    "Ethiopia": 0.514271,
    "Nigeria": 0.511785,
    "Pakistan": 0.514583,
}

MATERNAL_HEMORRHAGE_HEMOGLOBIN_POSTPARTUM_SHIFT = 6.8  # g/L
PROBABILITY_MODERATE_MATERNAL_HEMORRHAGE = (0.85, 0.81, 0.89)

# state: (mean_params, sd_params)
HEMOGLOBIN_CORRECTION_FACTORS = {
    models.NOT_PREGNANT_STATE_NAME: ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    models.PREGNANT_STATE_NAME: (
        (0.919325, 0.86, 0.98),
        (1.032920188, 1.032920188, 1.032920188),
    ),
}


class _HemoglobinDistributionParameters(NamedTuple):
    XMAX: int = 220
    EULERS_CONSTANT: float = np.euler_gamma
    GAMMA_DISTRIBUTION_WEIGHT: float = 0.4
    MIRROR_GUMBEL_DISTRIBUTION_WEIGHT: float = 0.6


HEMOGLOBIN_DISTRIBUTION_PARAMETERS = _HemoglobinDistributionParameters()

ANEMIA_DISABILITY_WEIGHTS = {
    "none": 0.0,
    "mild": 0.004,
    "moderate": 0.052,
    "severe": 0.149,
}

# tuples are: (age_start, age_end, severe_upper, moderate_upper, mild_upper)
_hemoglobin_threshold_data = {
    "pregnant": [(5, 15, 80, 110, 115), (15, 57, 70, 100, 110)],
    "not_pregnant": [(5, 15, 80, 110, 115), (15, 57, 80, 110, 120)],
}
_hemoglobin_state_map = {
    "pregnant": models.PREGNANCY_MODEL_STATES[1:],
    "not_pregnant": [models.NOT_PREGNANT_STATE_NAME],
}
_htd = []
for key, states in _hemoglobin_state_map.items():
    for state in states:
        for row in _hemoglobin_threshold_data[key]:
            _htd.append((state, "Female", *row))

HEMOGLOBIN_THRESHOLD_DATA = pd.DataFrame(
    _htd,
    columns=[
        "pregnancy_status",
        "sex",
        "age_start",
        "age_end",
        "severe",
        "moderate",
        "mild",
    ],
)

MATERNAL_BMI_ANEMIA_THRESHOLD = 100.0  # g/L, units of hemoglobin exposure distribution

SEVERE_ANEMIA_AMONG_PREGNANT_WOMEN_THRESHOLD = 70.0  # g/L


# Risk Effects
RR_MATERNAL_HEMORRHAGE_ATTRIBUTABLE_TO_HEMOGLOBIN = (
    3.54,
    1.2,
    10.4,
)  # (median, lower, upper) 95% CI
HEMOGLOBIN_SCALE_FACTOR_MODERATE_HEMORRHAGE = 0.9
HEMOGLOBIN_SCALE_FACTOR_SEVERE_HEMORRHAGE = 0.833

TMREL_HEMOGLOBIN_ON_MATERNAL_DISORDERS = 120.0
RR_SCALAR = (
    10.0  # Conversion factor between hemoglobin units (g/L) and relative risk units (g/dL)
)

RR_STILLBIRTH_PROBABILITY_ATTRIBUTABLE_TO_HEMOGLOBIN = (
    3.87,
    1.88,
    8.06,
)  # (median, lower, upper) 95% CI
