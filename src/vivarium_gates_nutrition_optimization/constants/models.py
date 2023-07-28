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
PARTURITION_STATE_NAME = "parturition"
POSTPARTUM_STATE_NAME = "postpartum"
PREGNANCY_MODEL_STATES = (
    NOT_PREGNANT_STATE_NAME,
    PREGNANT_STATE_NAME,
    PARTURITION_STATE_NAME,
    POSTPARTUM_STATE_NAME,
)
PREGNANCY_MODEL_TRANSITIONS = (
    TransitionString(f"{PREGNANT_STATE_NAME}_TO_{PARTURITION_STATE_NAME}"),
    TransitionString(f"{PARTURITION_STATE_NAME}_TO_{POSTPARTUM_STATE_NAME}"),
    TransitionString(f"{POSTPARTUM_STATE_NAME}_TO_{NOT_PREGNANT_STATE_NAME}"),
)
PARTIAL_TERM_OUTCOME = "partial_term"
LIVE_BIRTH_OUTCOME = "live_birth"
STILLBIRTH_OUTCOME = "stillbirth"
INVALID_OUTCOME = "invalid"  ## For sex of partial births

PREGNANCY_OUTCOMES = (
    PARTIAL_TERM_OUTCOME,
    LIVE_BIRTH_OUTCOME,
    STILLBIRTH_OUTCOME,
)

###########################
# Maternal Disorders      #
###########################

MATERNAL_DISORDERS_MODEL_NAME = data_keys.MATERNAL_DISORDERS.name
MATERNAL_DISORDERS_SUSCEPTIBLE_STATE_NAME = f"susceptible_to_{MATERNAL_DISORDERS_MODEL_NAME}"
MATERNAL_DISORDERS_STATE_NAME = MATERNAL_DISORDERS_MODEL_NAME
MATERNAL_DISORDERS_RECOVERED_STATE_NAME = f"recovered_from_{MATERNAL_DISORDERS_MODEL_NAME}"

MATERNAL_DISORDERS_MODEL_STATES = (
    MATERNAL_DISORDERS_SUSCEPTIBLE_STATE_NAME,
    MATERNAL_DISORDERS_STATE_NAME,
    MATERNAL_DISORDERS_RECOVERED_STATE_NAME,
)

MATERNAL_DISORDERS_MODEL_TRANSITIONS = (
    TransitionString(
        f"{MATERNAL_DISORDERS_SUSCEPTIBLE_STATE_NAME}_TO_{MATERNAL_DISORDERS_STATE_NAME}"
    ),
    TransitionString(
        f"{MATERNAL_DISORDERS_STATE_NAME}_TO_{MATERNAL_DISORDERS_RECOVERED_STATE_NAME}"
    ),
)


###########################
# Maternal Hemorrhage     #
###########################

MATERNAL_HEMORRHAGE_MODEL_NAME = data_keys.MATERNAL_HEMORRHAGE.name
MATERNAL_HEMORRHAGE_SUSCEPTIBLE_STATE_NAME = (
    f"susceptible_to_{MATERNAL_HEMORRHAGE_MODEL_NAME}"
)
MATERNAL_HEMORRHAGE_STATE_NAME = MATERNAL_HEMORRHAGE_MODEL_NAME
MATERNAL_HEMORRHAGE_RECOVERED_STATE_NAME = f"recovered_from_{MATERNAL_HEMORRHAGE_MODEL_NAME}"

MATERNAL_HEMORRHAGE_MODEL_STATES = (
    MATERNAL_HEMORRHAGE_SUSCEPTIBLE_STATE_NAME,
    MATERNAL_HEMORRHAGE_STATE_NAME,
    MATERNAL_HEMORRHAGE_RECOVERED_STATE_NAME,
)

MATERNAL_HEMORRHAGE_MODEL_TRANSITIONS = (
    TransitionString(
        f"{MATERNAL_HEMORRHAGE_SUSCEPTIBLE_STATE_NAME}_TO_{MATERNAL_HEMORRHAGE_STATE_NAME}"
    ),
    TransitionString(
        f"{MATERNAL_HEMORRHAGE_STATE_NAME}_TO_{MATERNAL_HEMORRHAGE_RECOVERED_STATE_NAME}"
    ),
)


STATE_MACHINE_MAP = {
    PREGNANCY_MODEL_NAME: {
        "states": PREGNANCY_MODEL_STATES,
        "transitions": PREGNANCY_MODEL_TRANSITIONS,
    },
    MATERNAL_DISORDERS_MODEL_NAME: {
        "states": MATERNAL_DISORDERS_MODEL_STATES,
        "transitions": MATERNAL_DISORDERS_MODEL_TRANSITIONS,
    },
    MATERNAL_HEMORRHAGE_MODEL_NAME: {
        "states": MATERNAL_HEMORRHAGE_MODEL_STATES,
        "transitions": MATERNAL_HEMORRHAGE_MODEL_TRANSITIONS,
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
