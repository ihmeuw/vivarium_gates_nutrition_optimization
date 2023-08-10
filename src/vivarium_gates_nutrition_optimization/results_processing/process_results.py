from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, Union

import pandas as pd
import yaml
from loguru import logger

from vivarium_gates_nutrition_optimization.constants import results, scenarios

SCENARIO_COLUMN = "scenario"
GROUPBY_COLUMNS = [results.INPUT_DRAW_COLUMN, SCENARIO_COLUMN]
OUTPUT_COLUMN_SORT_ORDER = [
    "age_group",
    "risk",
    "cause",
    "measure",
    "input_draw",
]
RENAME_COLUMNS = {
    "age_group": "age",
    "cause_of_death": "cause",
}


def make_measure_data(data):
    measure_data = MeasureData(
        ylls=get_by_cause_measure_data(data, "ylls"),
        ylds=get_by_cause_measure_data(data, "ylds"),
        deaths=get_by_cause_measure_data(data, "deaths"),
        pregnancy_state_person_time=get_state_person_time_measure_data(
            data, "pregnancy_state_person_time"
        ),
        pregnancy_transition_count=get_transition_count_measure_data(
            data, "pregnancy_transition_count"
        ),
        maternal_disorders_state_person_time=get_state_person_time_measure_data(
            data, "maternal_disorders_state_person_time"
        ),
        maternal_disorders_transition_count=get_transition_count_measure_data(
            data, "maternal_disorders_transition_count"
        ),
        maternal_hemorrhage_state_person_time=get_state_person_time_measure_data(
            data, "maternal_hemorrhage_state_person_time"
        ),
        maternal_hemorrhage_transition_count=get_transition_count_measure_data(
            data, "maternal_hemorrhage_transition_count"
        ),
        anemia_state_person_time=get_state_person_time_measure_data(
            data, "anemia_state_person_time"
        ),
        maternal_bmi_state_person_time=get_state_person_time_measure_data(
            data, "maternal_bmi_state_person_time"
        ),
    )
    return measure_data


class MeasureData(NamedTuple):
    ylls: pd.DataFrame
    ylds: pd.DataFrame
    deaths: pd.DataFrame
    pregnancy_state_person_time: pd.DataFrame
    pregnancy_transition_count: pd.DataFrame
    maternal_disorders_state_person_time: pd.DataFrame
    maternal_disorders_transition_count: pd.DataFrame
    maternal_hemorrhage_state_person_time: pd.DataFrame
    maternal_hemorrhage_transition_count: pd.DataFrame
    anemia_state_person_time: pd.DataFrame

    def dump(self, output_dir: Path):
        for key, df in self._asdict().items():
            df.to_hdf(output_dir / f"{key}.hdf", key=key)
            df.to_csv(output_dir / f"{key}.csv")


def read_data(path: Path, single_run: bool) -> Tuple[pd.DataFrame, List[str]]:
    data = pd.read_hdf(path)
    # noinspection PyUnresolvedReferences
    data = (
        data.drop(columns=data.columns.intersection(results.THROWAWAY_COLUMNS))
        .reset_index(drop=True)
        .rename(
            columns={
                results.OUTPUT_SCENARIO_COLUMN: SCENARIO_COLUMN,
                results.OUTPUT_INPUT_DRAW_COLUMN: results.INPUT_DRAW_COLUMN,
                results.OUTPUT_RANDOM_SEED_COLUMN: results.RANDOM_SEED_COLUMN,
            }
        )
    )
    if single_run:
        data[results.INPUT_DRAW_COLUMN] = 0
        data[results.RANDOM_SEED_COLUMN] = 0
        data[SCENARIO_COLUMN] = scenarios.INTERVENTION_SCENARIOS.BASELINE.name
        keyspace = {
            results.INPUT_DRAW_COLUMN: [0],
            results.RANDOM_SEED_COLUMN: [0],
            results.OUTPUT_SCENARIO_COLUMN: [scenarios.INTERVENTION_SCENARIOS.BASELINE.name],
        }
    else:
        data[results.INPUT_DRAW_COLUMN] = data[results.INPUT_DRAW_COLUMN].astype(int)
        data[results.RANDOM_SEED_COLUMN] = data[results.RANDOM_SEED_COLUMN].astype(int)
        with (path.parent / "keyspace.yaml").open() as f:
            keyspace = yaml.full_load(f)
    return data, keyspace


def filter_out_incomplete(
    data: pd.DataFrame, keyspace: Dict[str, Union[str, int]]
) -> pd.DataFrame:
    output = []
    for draw in keyspace[results.INPUT_DRAW_COLUMN]:
        # For each draw, gather all random seeds completed for all scenarios.
        random_seeds = set(keyspace[results.RANDOM_SEED_COLUMN])
        draw_data = data.loc[data[results.INPUT_DRAW_COLUMN] == draw]
        for scenario in keyspace[results.OUTPUT_SCENARIO_COLUMN]:
            seeds_in_data = draw_data.loc[
                data[SCENARIO_COLUMN] == scenario, results.RANDOM_SEED_COLUMN
            ].unique()
            random_seeds = random_seeds.intersection(seeds_in_data)
        draw_data = draw_data.loc[draw_data[results.RANDOM_SEED_COLUMN].isin(random_seeds)]
        output.append(draw_data)
    return pd.concat(output, ignore_index=True).reset_index(drop=True)


def aggregate_over_seed(data: pd.DataFrame) -> pd.DataFrame:
    non_count_columns = []
    for non_count_template in results.NON_COUNT_TEMPLATES:
        non_count_columns += results.RESULT_COLUMNS(non_count_template)
    count_columns = [c for c in data.columns if c not in non_count_columns + GROUPBY_COLUMNS]

    # non_count_data = data[non_count_columns + GROUPBY_COLUMNS].groupby(GROUPBY_COLUMNS).mean()
    count_data = data[count_columns + GROUPBY_COLUMNS].groupby(GROUPBY_COLUMNS).sum()
    return pd.concat(
        [
            count_data,
            # non_count_data
        ],
        axis=1,
    ).reset_index()


def pivot_data(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.set_index(GROUPBY_COLUMNS)
        .stack()
        .reset_index()
        .rename(columns={f"level_{len(GROUPBY_COLUMNS)}": "key", 0: "value"})
    )


def sort_data(data: pd.DataFrame) -> pd.DataFrame:
    sort_order = [c for c in OUTPUT_COLUMN_SORT_ORDER if c in data.columns]
    other_cols = [c for c in data.columns if c not in sort_order and c != "value"]
    data = data[sort_order + other_cols + ["value"]].sort_values(sort_order)
    return data.reset_index(drop=True)


def apply_results_map(data: pd.DataFrame, kind: str) -> pd.DataFrame:
    logger.info(f"Mapping {kind} data to stratifications.")
    map_df = results.RESULTS_MAP(kind)
    data = data.set_index("key")
    data = data.join(map_df).reset_index(drop=True)
    data = data.rename(columns=RENAME_COLUMNS)
    logger.info(f"Mapping {kind} complete.")
    return data


def get_measure_data(data: pd.DataFrame, measure: str) -> pd.DataFrame:
    data = pivot_data(data[results.RESULT_COLUMNS(measure) + GROUPBY_COLUMNS])
    data = apply_results_map(data, measure)
    return sort_data(data)


def get_by_cause_measure_data(data: pd.DataFrame, measure: str) -> pd.DataFrame:
    data = get_measure_data(data, measure)
    return sort_data(data)


def get_state_person_time_measure_data(data: pd.DataFrame, measure: str) -> pd.DataFrame:
    data = get_measure_data(data, measure)
    return sort_data(data)


def get_transition_count_measure_data(data: pd.DataFrame, measure: str) -> pd.DataFrame:
    # Oops, edge case.
    data = data.drop(columns=[c for c in data.columns if "event_count" in c and "2041" in c])
    data = get_measure_data(data, measure)
    return sort_data(data)
