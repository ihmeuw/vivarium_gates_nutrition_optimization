from typing import NamedTuple


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
