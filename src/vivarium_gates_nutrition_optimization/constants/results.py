import itertools

import pandas as pd

from vivarium_gates_nutrition_optimization.constants import models

#################################
# Results columns and variables #
#################################

TOTAL_YLDS_COLUMN = "years_lived_with_disability"
TOTAL_YLLS_COLUMN = "years_of_life_lost"

# Columns from parallel runs
INPUT_DRAW_COLUMN = "input_draw"
RANDOM_SEED_COLUMN = "random_seed"

OUTPUT_INPUT_DRAW_COLUMN = "input_data.input_draw_number"
OUTPUT_RANDOM_SEED_COLUMN = "randomness.random_seed"
OUTPUT_SCENARIO_COLUMN = "placeholder_branch_name.scenario"

STANDARD_COLUMNS = {
    "total_ylls": TOTAL_YLLS_COLUMN,
    "total_ylds": TOTAL_YLDS_COLUMN,
}

THROWAWAY_COLUMNS = [f"{state}_event_count" for state in models.STATES]

DEATH_COLUMN_TEMPLATE = "death_due_to_{CAUSE_OF_DEATH}_age_{AGE_GROUP}"
YLLS_COLUMN_TEMPLATE = "ylls_due_to_{CAUSE_OF_DEATH}_age_{AGE_GROUP}"
YLDS_COLUMN_TEMPLATE = "ylds_due_to_{CAUSE_OF_DISABILITY}_age_{AGE_GROUP}"
STATE_PERSON_TIME_COLUMN_TEMPLATE = "{STATE}_person_time_age_{AGE_GROUP}"
TRANSITION_COUNT_COLUMN_TEMPLATE = "{TRANSITION}_event_count_age_{AGE_GROUP}"
PREGNANCY_OUTPUT_COUNT_COLUMN_TEMPLATE = "outcome_{PREGNANCY_OUTCOME}_count_age_{AGE_GROUP}"

COLUMN_TEMPLATES = {
    "deaths": DEATH_COLUMN_TEMPLATE,
    "ylls": YLLS_COLUMN_TEMPLATE,
    "ylds": YLDS_COLUMN_TEMPLATE,
    "state_person_time": STATE_PERSON_TIME_COLUMN_TEMPLATE,
    "transition_count": TRANSITION_COUNT_COLUMN_TEMPLATE,
    "pregnancy_outcome_count": PREGNANCY_OUTPUT_COUNT_COLUMN_TEMPLATE,
}

NON_COUNT_TEMPLATES = []

AGE_GROUPS = (
    "10_to_14",
    "15_to_19",
    "20_to_24",
    "25_to_29",
    "30_to_34",
    "35_to_39",
    "40_to_44",
    "45_to_49",
    "50_to_54",
    "55_to_59",
)
# TODO - add causes of death
CAUSES_OF_DEATH = (
    "other_causes",
    # models.FIRST_STATE_NAME,
)
# TODO - add causes of disability
CAUSES_OF_DISABILITY = (
    # models.FIRST_STATE_NAME,
    # models.SECOND_STATE_NAME,
)

TEMPLATE_FIELD_MAP = {
    "AGE_GROUP": AGE_GROUPS,
    "CAUSE_OF_DEATH": CAUSES_OF_DEATH,
    "CAUSE_OF_DISABILITY": CAUSES_OF_DISABILITY,
    "STATE": models.STATES,
    "TRANSITION": models.TRANSITIONS,
    "PREGNANCY_OUTCOME": models.PREGNANCY_OUTCOMES,
}


# noinspection PyPep8Naming
def RESULT_COLUMNS(kind="all"):
    if kind not in COLUMN_TEMPLATES and kind != "all":
        raise ValueError(f"Unknown result column type {kind}")
    columns = []
    if kind == "all":
        for k in COLUMN_TEMPLATES:
            columns += RESULT_COLUMNS(k)
        columns = list(STANDARD_COLUMNS.values()) + columns
    else:
        template = COLUMN_TEMPLATES[kind]
        filtered_field_map = {
            field: values
            for field, values in TEMPLATE_FIELD_MAP.items()
            if f"{{{field}}}" in template
        }
        fields, value_groups = filtered_field_map.keys(), itertools.product(
            *filtered_field_map.values()
        )
        for value_group in value_groups:
            columns.append(
                template.format(**{field: value for field, value in zip(fields, value_group)})
            )
    return columns


# noinspection PyPep8Naming
def RESULTS_MAP(kind):
    if kind not in COLUMN_TEMPLATES:
        raise ValueError(f"Unknown result column type {kind}")
    columns = []
    template = COLUMN_TEMPLATES[kind]
    filtered_field_map = {
        field: values
        for field, values in TEMPLATE_FIELD_MAP.items()
        if f"{{{field}}}" in template
    }
    fields, value_groups = list(filtered_field_map.keys()), list(
        itertools.product(*filtered_field_map.values())
    )
    for value_group in value_groups:
        columns.append(
            template.format(**{field: value for field, value in zip(fields, value_group)})
        )
    df = pd.DataFrame(value_groups, columns=map(lambda x: x.lower(), fields))
    df["key"] = columns
    df[
        "measure"
    ] = kind  # per researcher feedback, this column is useful, even when it's identical for all rows
    return df.set_index("key").sort_index()
