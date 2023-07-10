from typing import NamedTuple

from vivarium_public_health.utilities import TargetString

#############
# Data Keys #
#############

METADATA_LOCATIONS = "metadata.locations"


class __Population(NamedTuple):
    LOCATION: str = "population.location"
    STRUCTURE: str = "population.structure"
    AGE_BINS: str = "population.age_bins"
    DEMOGRAPHY: str = "population.demographic_dimensions"
    TMRLE: str = "population.theoretical_minimum_risk_life_expectancy"
    ACMR: str = "cause.all_causes.cause_specific_mortality_rate"

    @property
    def name(self):
        return "population"

    @property
    def log_name(self):
        return "population"


POPULATION = __Population()


class __Pregnancy(NamedTuple):
    ASFR: str = "covariate.age_specific_fertility_rate.estimate"
    SBR: str = "covariate.stillbirth_to_live_birth_ratio.estimate"
    RAW_INCIDENCE_RATE_MISCARRIAGE: str = (
        "cause.maternal_abortion_and_miscarriage.raw_incidence_rate"
    )
    RAW_INCIDENCE_RATE_ECTOPIC: str = "cause.ectopic_pregnancy.raw_incidence_rate"

    @property
    def name(self):
        return "pregnancy"

    @property
    def log_name(self):
        return self.name.replace("_", " ")


PREGNANCY = __Pregnancy()


class __LowBirthWeightShortGestation(NamedTuple):
    # Keys that will be loaded into the artifact. must have a colon type declaration
    EXPOSURE: str = "risk_factor.low_birth_weight_and_short_gestation.exposure"
    DISTRIBUTION: str = "risk_factor.low_birth_weight_and_short_gestation.distribution"
    CATEGORIES: str = "risk_factor.low_birth_weight_and_short_gestation.categories"

    @property
    def name(self):
        return "low_birth_weight_and_short_gestation"

    @property
    def log_name(self):
        return "low birth weight and short gestation"


LBWSG = __LowBirthWeightShortGestation()

class __MaternalDisorders(NamedTuple):
    TOTAL_CSMR: str = "cause.maternal_disorders.cause_specific_mortality_rate"
    TOTAL_INCIDENCE_RATE: str = "cause.maternal_disorders.incidence_rate"
    HEMORRHAGE_CSMR: str = "cause.maternal_hemorrhage.cause_specific_mortality_rate"
    HEMORRHAGE_INCIDENCE_RATE: str = "cause.maternal_hemorrhage.incidence_rate"
    YLDS: str = "cause.maternal_disorders.ylds"

    PROBABILITY_FATAL: str = "covariate.probability_fatal_maternal_disorder.estimate"
    PROBABILITY_NONFATAL: str = "covariate.probability_nonfatal_maternal_disorder.estimate"
    PROBABILITY_HEMORRHAGE: str = "covariate.probability_maternal_hemorrhage.estimate"

    @property
    def name(self):
        return "maternal_disorders"

    @property
    def log_name(self):
        return "maternal_disorders"

MATERNAL_DISORDERS = __MaternalDisorders()

MAKE_ARTIFACT_KEY_GROUPS = [POPULATION, PREGNANCY, LBWSG, MATERNAL_DISORDERS]
