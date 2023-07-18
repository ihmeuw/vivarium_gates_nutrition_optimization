from vivarium.framework.engine import Builder
from vivarium.framework.values import Pipeline
from vivarium_public_health.population.mortality import Mortality as Mortality_


class Mortality(Mortality_):
    def _get_mortality_rate(self, builder: Builder) -> Pipeline:
        ## We want to register a value producer instead of a rate producer, so the
        ## Rescaling doesn't happen.
        return builder.value.register_value_producer(
            self.mortality_rate_pipeline_name,
            source=self._calculate_mortality_rate,
            requires_columns=["age", "sex"],
        )
