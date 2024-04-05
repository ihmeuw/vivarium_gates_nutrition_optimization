import itertools

import pandas as pd

from vivarium_gates_nutrition_optimization.constants import data_values, models

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
OUTPUT_SCENARIO_COLUMN = "intervention.scenario"

STANDARD_COLUMNS = {
    "total_ylls": TOTAL_YLLS_COLUMN,
    "total_ylds": TOTAL_YLDS_COLUMN,
}

THROWAWAY_COLUMNS = [f"{state}_event_count" for state in models.STATES]

DEATH_COLUMN_TEMPLATE = "MEASURE_death_due_to_{CAUSE_OF_DEATH}_AGE_GROUP_{AGE_GROUP}"
YLLS_COLUMN_TEMPLATE = "MEASURE_ylls_due_to_{CAUSE_OF_DEATH}_AGE_GROUP_{AGE_GROUP}"
YLDS_COLUMN_TEMPLATE = "MEASURE_ylds_due_to_{CAUSE_OF_DISABILITY}_AGE_GROUP_{AGE_GROUP}"
PREGNANCY_STATE_PERSON_TIME_COLUMN_TEMPLATE = (
    "MEASURE_{PREGNANCY_STATE}_person_time_AGE_GROUP_{AGE_GROUP}"
)
PREGNANCY_TRANSITION_COUNT_COLUMN_TEMPLATE = (
    "MEASURE_{PREGNANCY_TRANSITION}_event_count_AGE_GROUP_{AGE_GROUP}"
)

MATERNAL_DISORDERS_PERSON_TIME_COLUMN_TEMPLATE = (
    "MEASURE_{MATERNAL_DISORDERS_STATE}_person_time_AGE_GROUP_{AGE_GROUP}"
)
MATERNAL_DISORDERS_TRANSITION_COUNT_COLUMN_TEMPLATE = (
    "MEASURE_{MATERNAL_DISORDERS_TRANSITION}_event_count_AGE_GROUP_{AGE_GROUP}"
)

MATERNAL_HEMORRHAGE_PERSON_TIME_COLUMN_TEMPLATE = (
    "MEASURE_{MATERNAL_HEMORRHAGE_STATE}_person_time_AGE_GROUP_{AGE_GROUP}"
)
MATERNAL_HEMORRHAGE_TRANSITION_COUNT_COLUMN_TEMPLATE = (
    "MEASURE_{MATERNAL_HEMORRHAGE_TRANSITION}_event_count_AGE_GROUP_{AGE_GROUP}"
)

ANEMIA_STATE_PERSON_TIME_COLUMN_TEMPLATE = (
    "MEASURE_anemia_{ANEMIA_LEVEL}_person_time_AGE_GROUP_{AGE_GROUP}"
)

MATERNAL_BMI_ANEMIA_PERSON_TIME_COLUMN_TEMPLATE = "MEASURE_maternal_bmi_anemia_{MATERNAL_BMI_ANEMIA_CATEGORY}_person_time_AGE_GROUP_{AGE_GROUP}"

INTERVENTION_COUNT_COLUMN_TEMPLATE = (
    "MEASURE_intervention_{INTERVENTION}_count_AGE_GROUP_{AGE_GROUP}"
)

PREGNANCY_OUTCOME_COUNT_COLUMN_TEMPLATE = (
    "MEASURE_pregnancy_outcome_{PREGNANCY_OUTCOME}_count_AGE_GROUP_{AGE_GROUP}"
)

COLUMN_TEMPLATES = {
    "deaths": DEATH_COLUMN_TEMPLATE,
    "ylls": YLLS_COLUMN_TEMPLATE,
    "ylds": YLDS_COLUMN_TEMPLATE,
    "pregnancy_state_person_time": PREGNANCY_STATE_PERSON_TIME_COLUMN_TEMPLATE,
    "pregnancy_transition_count": PREGNANCY_TRANSITION_COUNT_COLUMN_TEMPLATE,
    "maternal_disorders_state_person_time": MATERNAL_DISORDERS_PERSON_TIME_COLUMN_TEMPLATE,
    "maternal_disorders_transition_count": MATERNAL_DISORDERS_TRANSITION_COUNT_COLUMN_TEMPLATE,
    "maternal_hemorrhage_state_person_time": MATERNAL_HEMORRHAGE_PERSON_TIME_COLUMN_TEMPLATE,
    "maternal_hemorrhage_transition_count": MATERNAL_HEMORRHAGE_TRANSITION_COUNT_COLUMN_TEMPLATE,
    "anemia_state_person_time": ANEMIA_STATE_PERSON_TIME_COLUMN_TEMPLATE,
    "maternal_bmi_anemia_person_time": MATERNAL_BMI_ANEMIA_PERSON_TIME_COLUMN_TEMPLATE,
    "intervention_count": INTERVENTION_COUNT_COLUMN_TEMPLATE,
    "pregnancy_outcome_count": PREGNANCY_OUTCOME_COUNT_COLUMN_TEMPLATE,
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

CAUSES_OF_DEATH = ("maternal_disorders",)
CAUSES_OF_DISABILITY = ("maternal_disorders", "anemia", "all_causes")  # "other_causes",

TEMPLATE_FIELD_MAP = {
    "AGE_GROUP": AGE_GROUPS,
    "CAUSE_OF_DEATH": CAUSES_OF_DEATH,
    "CAUSE_OF_DISABILITY": CAUSES_OF_DISABILITY,
    "PREGNANCY_STATE": models.PREGNANCY_MODEL_STATES,
    "PREGNANCY_TRANSITION": models.PREGNANCY_MODEL_TRANSITIONS,
    "PREGNANCY_OUTCOME": models.PREGNANCY_OUTCOMES,
    "MATERNAL_DISORDERS_STATE": models.MATERNAL_DISORDERS_MODEL_STATES,
    "MATERNAL_DISORDERS_TRANSITION": models.MATERNAL_DISORDERS_MODEL_TRANSITIONS,
    "MATERNAL_HEMORRHAGE_STATE": models.MATERNAL_HEMORRHAGE_MODEL_STATES,
    "MATERNAL_HEMORRHAGE_TRANSITION": models.MATERNAL_HEMORRHAGE_MODEL_TRANSITIONS,
    "ANEMIA_LEVEL": data_values.ANEMIA_DISABILITY_WEIGHTS.keys(),
    "ANEMIA_STATUS_AT_BIRTH": data_values.ANEMIA_STATUS_AT_BIRTH_CATEGORIES,
    "MATERNAL_BMI_ANEMIA_CATEGORY": models.BMI_ANEMIA_CATEGORIES,
    "INTERVENTION": models.SUPPLEMENTATION_CATEGORIES,
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
