from pathlib import Path

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium_cluster_tools.utilities import mkdir

from vivarium_gates_nutrition_optimization.constants import models


class BirthRecorder:
    @property
    def name(self):
        return "birth_recorder"

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        self.output_path = self._build_output_path(builder)
        self.randomness = builder.randomness.get_stream(self.name)

        self.births = []

        required_columns = [
            "pregnancy_term_outcome",
            "pregnancy_duration",
            "pregnancy",
            "previous_pregnancy",
        ]
        self.population_view = builder.population.get_view(required_columns)

        builder.event.register_listener("collect_metrics", self.on_collect_metrics)
        builder.event.register_listener("simulation_end", self.write_output)

    def on_collect_metrics(self, event: Event):
        pop = self.population_view.get(event.index)
        new_birth_mask = (
            (pop["pregnancy_term_outcome"] == models.FULL_TERM_OUTCOME)
            & (pop["previous_pregnancy"] == models.PREGNANT_STATE_NAME)
            & (pop["pregnancy"] == models.POSTPARTUM_STATE_NAME)
        )

        new_births = pop.loc[new_birth_mask, ["pregnancy_duration"]]

        self.births.append(new_births)

    # noinspection PyUnusedLocal
    def write_output(self, event: Event) -> None:
        births_data = pd.concat(self.births)
        births_data.to_hdf(f"{self.output_path}.hdf", key="data")
        births_data.to_csv(f"{self.output_path}.csv")

    ###########
    # Helpers #
    ###########

    @staticmethod
    def _build_output_path(builder: Builder) -> Path:
        results_root = builder.configuration.output_data.results_directory
        output_root = Path(results_root) / "child_data"

        mkdir(output_root, exists_ok=True)

        input_draw = builder.configuration.input_data.input_draw_number
        seed = builder.configuration.randomness.random_seed
        output_path = output_root / f"draw_{input_draw}_seed_{seed}"

        return output_path
