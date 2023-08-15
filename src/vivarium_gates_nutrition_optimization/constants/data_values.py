from typing import NamedTuple

import numpy as np
import pandas as pd

from vivarium_gates_nutrition_optimization.constants import data_keys, models, paths


############################
# Disease Model Parameters #
############################
class _Durations(NamedTuple):
    FULL_TERM_DAYS = 40 * 7
    POSTPARTUM_DAYS = 6 * 7
    PARTURITION_DAYS = 1 * 7
    DETECTION_DAYS = 6 * 7
    PARTIAL_TERM_DAYS = 24 * 7
    INTERVENTION_DELAY_DAYS = 8 * 7


DURATIONS = _Durations()


INFANT_MALE_PERCENTAGES = {
    "Ethiopia": 0.514271,
    "Nigeria": 0.511785,
    "Pakistan": 0.514583,
}

MATERNAL_HEMORRHAGE_HEMOGLOBIN_POSTPARTUM_SHIFT = 6.8  # g/L
PROBABILITY_MODERATE_MATERNAL_HEMORRHAGE = (0.85, 0.81, 0.89)


class _HemoglobinDistributionParameters(NamedTuple):
    XMAX: int = 220
    EULERS_CONSTANT: float = np.euler_gamma
    GAMMA_DISTRIBUTION_WEIGHT: float = 0.4
    MIRROR_GUMBEL_DISTRIBUTION_WEIGHT: float = 0.6


HEMOGLOBIN_DISTRIBUTION_PARAMETERS = _HemoglobinDistributionParameters()
ANEMIA_STATUS_AT_BIRTH_CATEGORIES = (
    "invalid",  ## Check on anemia_status that hasn't been properly assigned
    "not_anemic",
    "mild",
    "moderate",
    "severe",
)
ANEMIA_DISABILITY_WEIGHTS = {
    "not_anemic": 0.0,
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

ANEMIA_THRESHOLD_DATA = pd.DataFrame(
    _htd,
    columns=[
        "pregnancy",
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

PREGNANCY_CORRECTION_FACTORS = {
    data_keys.HEMOGLOBIN.MEAN: pd.read_csv(
        paths.HEMOGLOBIN_PREGNANCY_ADJUSTMENT_FACTORS_CSV, index_col=0
    ).squeeze(),
    data_keys.HEMOGLOBIN.STANDARD_DEVIATION: pd.Series(
        np.repeat(1.032920188, 1000), [f"draw_{i}" for i in range(1000)]
    ),
}
PROBABILITY_MODERATE_MATERNAL_HEMORRHAGE = (0.85, 0.81, 0.89)

RR_MATERNAL_HEMORRHAGE_ATTRIBUTABLE_TO_HEMOGLOBIN = (
    3.54,
    1.2,
    10.4,
)  # (median, lower, upper) 95% CI

INTERVENTION_SCENARIO_COVERAGE = pd.read_csv(
    paths.INTERVENTION_COVERAGE_BY_SCENARIO_CSV
).set_index("scenario")

_IFA_EFFECT_SIZE_LOWER = 4.08
_IFA_EFFECT_SIZE_UPPER = 11.52
_IFA_EFFECT_SIZE_SD = (_IFA_EFFECT_SIZE_UPPER - _IFA_EFFECT_SIZE_LOWER) / (2 * 1.96)  # 95% CI
IFA_EFFECT_SIZE = (7.8, _IFA_EFFECT_SIZE_SD)  # (mean, sd) g/L
