from vivarium_gates_nutrition_optimization.constants import data_keys


class TransitionString(str):
    def __new__(cls, value):
        # noinspection PyArgumentList
        obj = str.__new__(cls, value.lower())
        obj.from_state, obj.to_state = value.split("_TO_")
        return obj


###########################
# Pregnancy Model         #
###########################

PREGNANCY_MODEL_NAME = data_keys.PREGNANCY.name
PREGNANT_STATE_NAME = "pregnant"
NOT_PREGNANT_STATE_NAME = "not_pregnant"
POSTPARTUM_STATE_NAME = "postpartum"
PREGNANCY_MODEL_STATES = (
    NOT_PREGNANT_STATE_NAME,
    PREGNANT_STATE_NAME,
    POSTPARTUM_STATE_NAME,
)
PREGNANCY_MODEL_TRANSITIONS = (
    TransitionString(f"{PREGNANT_STATE_NAME}_TO_{POSTPARTUM_STATE_NAME}"),
    TransitionString(f"{POSTPARTUM_STATE_NAME}_TO_{NOT_PREGNANT_STATE_NAME}"),
)
FULL_TERM_OUTCOME = "full_term"
PARTIAL_TERM_OUTCOME = "partial_term"

PREGNANCY_TERM_OUTCOMES = (FULL_TERM_OUTCOME, PARTIAL_TERM_OUTCOME )

STATE_MACHINE_MAP = {
    PREGNANCY_MODEL_NAME: {
        "states": PREGNANCY_MODEL_STATES,
        "transitions": PREGNANCY_MODEL_TRANSITIONS,
    },
}


STATES = tuple(
    f"{model}_{state}"
    for model, state in STATE_MACHINE_MAP.items()
    for state in STATE_MACHINE_MAP[model]["states"]
)
TRANSITIONS = tuple(
    f"{model}_{transition}"
    for model, transition in STATE_MACHINE_MAP.items()
    for transition in STATE_MACHINE_MAP[model]["transitions"]
)
