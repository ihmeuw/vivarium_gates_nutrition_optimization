from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.lookup import LookupTable
from vivarium.framework.population import PopulationView, SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.state_machine import State, Transition
from vivarium.framework.values import Pipeline, list_combiner, union_post_processor
from vivarium_public_health.disease import DiseaseState as DiseaseState_
from vivarium_public_health.disease import SusceptibleState
from vivarium_public_health.disease.transition import ProportionTransition
from vivarium_public_health.utilities import is_non_zero

from vivarium_gates_nutrition_optimization.constants import data_keys, models


class DiseaseState(DiseaseState_):
    """State representing a disease in a state machine model."""

    def __init__(
        self,
        cause: str,
        get_data_functions: Dict[str, Callable] = None,
        cleanup_function: Callable = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        cause : str
            The name of this state.
        disability_weight : pandas.DataFrame or float, optional
            The amount of disability associated with this state.
        prevalence_data : pandas.DataFrame, optional
            The baseline occurrence of this state in a population.
        dwell_time : pandas.DataFrame or pandas.Timedelta, optional
            The minimum time a simulant exists in this state.
        event_time_column : str, optional
            The name of a column to track the last time this state was entered.
        event_count_column : str, optional
            The name of a column to track the number of times this state was entered.
        side_effect_function : callable, optional
            A function to be called when this state is entered.
        """
        super(DiseaseState_, self).__init__(cause, **kwargs)

        self.excess_mortality_rate_pipeline_name = f"{self.state_id}.excess_mortality_rate"
        self.excess_mortality_rate_paf_pipeline_name = (
            f"{self.excess_mortality_rate_pipeline_name}.paf"
        )

        self._get_data_functions = (
            get_data_functions if get_data_functions is not None else {}
        )
        self.cleanup_function = cleanup_function

        if self.cause is None and not set(self._get_data_functions.keys()).issuperset(
            ["disability_weight", "dwell_time", "prevalence"]
        ):
            raise ValueError(
                "If you do not provide a cause, you must supply"
                "custom data gathering functions for disability_weight, prevalence, and dwell_time."
            )

    #################
    # Setup methods #
    #################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super(DiseaseState_, self).setup(builder)
        self.prevalence = self.get_prevalence_table(builder)
        self.birth_prevalence = self.get_birth_prevalence_table(builder)
        self.dwell_time = self.get_dwell_time_pipeline(builder)
        self.has_disability, self.base_disability_weight = self.get_disability_weight_table(
            builder
        )
        self.disability_weight = self.get_disability_weight_pipeline(builder)
        self.register_disability_weight_modifier(builder)
        (
            self.has_excess_mortality,
            self.base_excess_mortality_rate,
        ) = self.get_excess_mortality_rate_table(builder)
        self.excess_mortality_rate = self.get_excess_mortality_rate_pipeline(builder)
        self.joint_paf = self.get_joint_paf_pipeline(builder)
        self.register_mortality_rate_modifier(builder)
        self.randomness_prevalence = self.get_prevalence_random_stream(builder)

    def get_prevalence_random_stream(self, builder: Builder) -> RandomnessStream:
        return builder.randomness.get_stream(f"{self.state_id}_prevalent_cases")

    ##################################
    # Lookup Tables                  #
    ##################################

    def get_prevalence_table(self, builder: Builder) -> LookupTable:
        prevalence_data = self.load_prevalence_data(builder)
        return builder.lookup.build_table(
            prevalence_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

    def get_birth_prevalence_table(self, builder: Builder) -> LookupTable:
        birth_prevalence_data = self.load_birth_prevalence_data(builder)
        return builder.lookup.build_table(
            birth_prevalence_data, key_columns=["sex"], parameter_columns=["year"]
        )

    def get_disability_weight_table(self, builder: Builder) -> Tuple[bool, LookupTable]:
        disability_weight_data = self.load_disability_weight_data(builder)
        has_disability = is_non_zero(disability_weight_data)
        base_disability_weight = builder.lookup.build_table(
            disability_weight_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )

        return has_disability, base_disability_weight

    def get_excess_mortality_rate_table(self, builder: Builder) -> Tuple[bool, LookupTable]:
        excess_mortality_data = self.load_excess_mortality_rate_data(builder)
        has_excess_mortality = is_non_zero(excess_mortality_data)
        base_excess_mortality_rate = builder.lookup.build_table(
            excess_mortality_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        return has_excess_mortality, base_excess_mortality_rate

    ##################################
    # Pipeline Sources               #
    ##################################

    def compute_disability_weight(self, index: pd.Index) -> pd.Series:
        """Gets the disability weight associated with this state.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.

        Returns
        -------
        `pandas.Series`
            An iterable of disability weights indexed by the provided `index`.
        """
        disability_weight = pd.Series(0, index=index)
        with_condition = self.with_condition(index)
        disability_weight.loc[with_condition] = self.base_disability_weight(with_condition)
        return disability_weight

    def compute_excess_mortality_rate(self, index: pd.Index) -> pd.Series:
        excess_mortality_rate = pd.Series(0, index=index)
        with_condition = self.with_condition(index)
        base_excess_mort = self.base_excess_mortality_rate(with_condition)
        joint_mediated_paf = self.joint_paf(with_condition)
        excess_mortality_rate.loc[with_condition] = base_excess_mort * (
            1 - joint_mediated_paf.values
        )
        return excess_mortality_rate

    def adjust_mortality_rate(self, index: pd.Index, rates_df: pd.DataFrame) -> pd.DataFrame:
        """Modifies the baseline mortality rate for a simulant if they are in this state.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.
        rates_df : `pandas.DataFrame`

        """
        rate = self.excess_mortality_rate(index, skip_post_processor=True)
        rates_df[self.state_id] = rate
        return rates_df

    ##################################
    # Pipelines                      #
    ##################################

    def get_dwell_time_pipeline(self, builder: Builder) -> Pipeline:
        dwell_time_data = self.load_dwell_time_data(builder)
        dwell_time_source = builder.lookup.build_table(
            dwell_time_data, key_columns=["sex"], parameter_columns=["age", "year"]
        )
        return builder.value.register_value_producer(
            f"{self.state_id}.dwell_time",
            source=dwell_time_source,
            requires_columns=["age", "sex"],
        )

    def get_disability_weight_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_value_producer(
            f"{self.state_id}.disability_weight",
            source=self.compute_disability_weight,
            requires_columns=["age", "sex", "alive", self._model],
        )

    def get_excess_mortality_rate_pipeline(self, builder: Builder) -> Pipeline:
        return builder.value.register_rate_producer(
            self.excess_mortality_rate_pipeline_name,
            source=self.compute_excess_mortality_rate,
            requires_columns=["age", "sex", "alive", self._model],
            requires_values=[self.excess_mortality_rate_paf_pipeline_name],
        )

    def get_joint_paf_pipeline(self, builder: Builder) -> Pipeline:
        paf = builder.lookup.build_table(0)
        return builder.value.register_value_producer(
            self.excess_mortality_rate_paf_pipeline_name,
            source=lambda idx: [paf(idx)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    ##################################
    # Pipeline Modifiers             #
    ##################################

    def register_disability_weight_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            "disability_weight", modifier=self.disability_weight
        )

    def register_mortality_rate_modifier(self, builder: Builder) -> None:
        builder.value.register_value_modifier(
            "mortality_rate",
            modifier=self.adjust_mortality_rate,
            requires_values=[self.excess_mortality_rate_pipeline_name],
        )

    ##################################
    # Data Loading Methods           #
    ##################################

    def load_prevalence_data(self, builder: Builder):
        if "prevalence" in self._get_data_functions:
            return self._get_data_functions["prevalence"](builder, self.cause)
        else:
            return builder.data.load(f"{self.cause_type}.{self.cause}.prevalence")

    def load_birth_prevalence_data(self, builder: Builder):
        if "birth_prevalence" in self._get_data_functions:
            return self._get_data_functions["birth_prevalence"](builder, self.cause)
        else:
            return 0

    def load_dwell_time_data(self, builder: Builder):
        if "dwell_time" in self._get_data_functions:
            dwell_time = self._get_data_functions["dwell_time"](builder, self.cause)
        else:
            dwell_time = 0

        if isinstance(dwell_time, pd.Timedelta):
            dwell_time = dwell_time.total_seconds() / (60 * 60 * 24)
        if (
            isinstance(dwell_time, pd.DataFrame) and np.any(dwell_time.value != 0)
        ) or dwell_time > 0:
            self.transition_set.allow_null_transition = True

        return dwell_time

    def load_disability_weight_data(self, builder: Builder):
        if "disability_weight" in self._get_data_functions:
            disability_weight = self._get_data_functions["disability_weight"](
                builder, self.cause
            )
        else:
            disability_weight = builder.data.load(
                f"{self.cause_type}.{self.cause}.disability_weight"
            )

        if isinstance(disability_weight, pd.DataFrame) and len(disability_weight) == 1:
            disability_weight = disability_weight.value[0]  # sequela only have single value

        return disability_weight

    def load_excess_mortality_rate_data(self, builder: Builder):
        if "excess_mortality_rate" in self._get_data_functions:
            return self._get_data_functions["excess_mortality_rate"](builder, self.cause)

        only_morbid = builder.data.load(f"cause.{self._model}.restrictions")["yld_only"]
        if only_morbid:
            return 0
        else:
            return builder.data.load(f"{self.cause_type}.{self.cause}.excess_mortality_rate")

    ########################
    # Event-driven methods #
    ########################

    def get_initial_event_times(self, pop_data: SimulantData) -> pd.DataFrame:
        pop_update = super(DiseaseState_, self).get_initial_event_times(pop_data)

        simulants_with_condition = self.population_view.subview([self._model]).get(
            pop_data.index, query=f'{self._model}=="{self.state_id}"'
        )
        if not simulants_with_condition.empty:
            infected_at = self._assign_event_time_for_prevalent_cases(
                simulants_with_condition,
                self.clock(),
                self.randomness_prevalence.get_draw,
                self.dwell_time,
            )
            pop_update.loc[infected_at.index, self.event_time_column] = infected_at

        return pop_update

    @staticmethod
    def _assign_event_time_for_prevalent_cases(
        infected: pd.Index,
        current_time: pd.Timestamp,
        randomness_func: Callable,
        dwell_time_func: Callable,
    ) -> pd.Timestamp:
        dwell_time = dwell_time_func(infected.index)
        infected_at = dwell_time * randomness_func(infected.index)
        infected_at = current_time - pd.to_timedelta(infected_at, unit="D")
        return infected_at

    def add_transition(
        self,
        output: State,
        source_data_type: str = None,
        get_data_functions: Dict[str, Callable] = None,
        **kwargs,
    ) -> Transition:
        if source_data_type == "rate":
            if get_data_functions is None:
                get_data_functions = {
                    "remission_rate": lambda builder, cause: builder.data.load(
                        f"{self.cause_type}.{cause}.remission_rate"
                    )
                }
            elif (
                "remission_rate" not in get_data_functions
                and "transition_rate" not in get_data_functions
            ):
                raise ValueError(
                    "You must supply a transition rate or remission rate function."
                )
        elif source_data_type == "proportion":
            if "proportion" not in get_data_functions:
                raise ValueError("You must supply a proportion function.")
        return super(DiseaseState_, self).add_transition(
            output, source_data_type, get_data_functions, **kwargs
        )

    def next_state(
        self, index: pd.Index, event_time: pd.Timestamp, population_view: PopulationView
    ) -> None:
        """Moves a population among different disease states.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.
        event_time:
            The time at which this transition occurs.
        population_view:
            A view of the internal state of the simulation.
        """
        eligible_index = self._filter_for_transition_eligibility(index, event_time)
        return super(DiseaseState_, self).next_state(
            eligible_index, event_time, population_view
        )

    def with_condition(self, index: pd.Index) -> pd.Index:
        pop = self.population_view.subview(["alive", self._model]).get(index)
        with_condition = pop.loc[
            (pop[self._model] == self.state_id) & (pop["alive"] == "alive")
        ].index
        return with_condition

    def _filter_for_transition_eligibility(
        self, index: pd.Index, event_time: pd.Timestamp
    ) -> pd.Index:
        """Filter out all simulants who haven't been in the state for the prescribed dwell time.

        Parameters
        ----------
        index
            An iterable of integer labels for the simulants.

        Returns
        -------
        pd.Index
            A filtered index of the simulants.
        """
        population = self.population_view.get(index, query='alive == "alive"')
        if np.any(self.dwell_time(index)) > 0:
            state_exit_time = population[self.event_time_column] + pd.to_timedelta(
                self.dwell_time(index), unit="D"
            )
            return population.loc[state_exit_time <= event_time].index
        else:
            return index

    def _cleanup_effect(self, index: pd.Index, event_time: pd.Timestamp) -> None:
        if self.cleanup_function is not None:
            self.cleanup_function(index, event_time)

    def __repr__(self) -> str:
        return "DiseaseState({})".format(self.state_id)


class ParturitionSelectionState(SusceptibleState):
    def add_transition(
        self,
        output: State,
        source_data_type: str = "proportion",
        get_data_functions: Dict[str, Callable] = None,
        **kwargs,
    ) -> Transition:
        transition = ParturitionSelectionTransition(
            self,
            output,
            get_data_functions={
                "proportion": lambda builder, cause: builder.data.load(
                    f"cause.{cause}.incident_probability"
                )
            },
            **kwargs,
        )
        self.transition_set.append(transition)
        return transition


class ParturitionSelectionTransition(ProportionTransition):
    def setup(self, builder: Builder):
        super().setup(builder)

        pipeline_name = f"{self.output_state.cause}.transition_proportion"
        self.proportion_pipeline = builder.value.register_value_producer(
            pipeline_name,
            source=self.compute_transition_proportion,
            requires_columns=["age", "sex", "alive"],
        )
        self.population_view = builder.population.get_view(["alive", "pregnancy"])

    def compute_transition_proportion(self, index):
        transition_proportion = pd.Series(0.0, index=index)
        sub_pop = self.population_view.get(
            index, query="(alive == 'alive') & (pregnancy == 'parturition')"
        ).index

        transition_proportion.loc[sub_pop] = self.proportion(sub_pop)
        return transition_proportion

    def _probability(self, index):
        return self.proportion_pipeline(index)

class ParturitionExclusionState(DiseaseState):
        def setup(self, builder: Builder):
            super().setup(builder)
            view_columns = self.columns_created + [self._model, "alive", "pregnancy", "tracked"]
            self.population_view = builder.population.get_view(view_columns)

        def get_disability_weight_pipeline(self, builder: Builder) -> Pipeline:
            return builder.value.register_value_producer(
            f"{self.state_id}.disability_weight",
            source=self.compute_disability_weight,
            requires_columns=["age", "sex", "alive", "pregnancy", self._model],
        )
        def compute_disability_weight(self, index: pd.Index):
            disability_weight = raw_disability_weight = pd.Series(0, index=index)
            with_condition = self.with_condition(index)
            raw_disability_weight.loc[with_condition] = self.base_disability_weight(with_condition)

            dw_map = {
                models.NOT_PREGNANT_STATE_NAME: raw_disability_weight,
                models.PREGNANT_STATE_NAME: raw_disability_weight,
                ## Pause YLD accumulation during the parturition state
                models.PARTURITION_STATE_NAME: pd.Series(0, index=index),
                models.POSTPARTUM_STATE_NAME: raw_disability_weight,
            }

            pop = self.population_view.get(index)
            alive = pop["alive"] == "alive"
            for state, dw in dw_map.items():
                in_state = alive & (pop["pregnancy"] == state)
                disability_weight[in_state] = dw.loc[in_state]

            return disability_weight