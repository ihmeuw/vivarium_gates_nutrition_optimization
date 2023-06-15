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
# Only one state for now for person-time
PREGNANCY_STATE = "pregnant"
PREGNANCY_SUSCEPTIBLE_STATE_NAME = "not_pregnant"
POSTPARTUM_STATE_NAME = "postpartum"
PREGNANCY_MODEL_STATES = (PREGNANCY_SUSCEPTIBLE_STATE_NAME, PREGNANCY_STATE, POSTPARTUM_STATE_NAME)
PREGNANCY_MODEL_TRANSITIONS = ()


###########################
# Disease Model variables #
###########################

# TODO input details of model states and transitions
SOME_MODEL_NAME = data_keys.SOME_DISEASE.name
SUSCEPTIBLE_STATE_NAME = f"susceptible_to_{SOME_MODEL_NAME}"
FIRST_STATE_NAME = "first_state"
SECOND_STATE_NAME = "second_state"
SOME_DISEASE_MODEL_STATES = (SUSCEPTIBLE_STATE_NAME, FIRST_STATE_NAME, SECOND_STATE_NAME)
SOME_DISEASE_MODEL_TRANSITIONS = (
    TransitionString(f"{SUSCEPTIBLE_STATE_NAME}_TO_{FIRST_STATE_NAME}"),
    TransitionString(f"{FIRST_STATE_NAME}_TO_{SECOND_STATE_NAME}"),
    TransitionString(f"{SECOND_STATE_NAME}_TO_{FIRST_STATE_NAME}"),
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
