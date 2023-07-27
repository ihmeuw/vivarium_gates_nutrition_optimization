from typing import NamedTuple

from vivarium_gates_nutrition_optimization.constants import data_keys


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
## mean, lower, upper to go into distribution
PREGNANCY_CORRECTION_FACTORS = {
    data_keys.HEMOGLOBIN.MEAN: (0.919325, 0.86, 0.98),
    data_keys.HEMOGLOBIN.STANDARD_DEVIATION: (1.032920188, 1.032920188, 1.032920188),
}
