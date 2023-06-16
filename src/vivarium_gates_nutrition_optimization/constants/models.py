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
PREGNANCY_SUSCEPTIBLE_STATE_NAME = "not_pregnant"
POSTPARTUM_STATE_NAME = "postpartum"
PREGNANCY_MODEL_STATES = (
    PREGNANCY_SUSCEPTIBLE_STATE_NAME,
    PREGNANT_STATE_NAME,
    POSTPARTUM_STATE_NAME,
)
PREGNANCY_MODEL_TRANSITIONS = (
    TransitionString(f"{PREGNANT_STATE_NAME}_TO_{POSTPARTUM_STATE_NAME}"),
)


STATE_MACHINE_MAP = {
    PREGNANCY_MODEL_NAME: {
        "states": PREGNANCY_MODEL_STATES,
        "transitions": PREGNANCY_MODEL_TRANSITIONS,
    },
}


STATES = tuple(state for model in STATE_MACHINE_MAP.values() for state in model["states"])
TRANSITIONS = tuple(
    state for model in STATE_MACHINE_MAP.values() for state in model["transitions"]
)
