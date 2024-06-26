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
    #    BACKGROUND_MORBIDITY: str = "cause.other_causes.disability_weight"

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
    RAW_INCIDENCE_RATE: str = "cause.maternal_disorders.raw_incidence_rate"
    INCIDENT_PROBABILITY: str = "cause.maternal_disorders.incident_probability"
    CSMR: str = "cause.maternal_disorders.cause_specific_mortality_rate"
    MORTALITY_PROBABILITY: str = "cause.maternal_disorders.mortality_probability"
    YLDS: str = "cause.maternal_disorders.ylds"

    RR_ATTRIBUTABLE_TO_HEMOGLOBIN: str = (
        "risk_factor.hemoglobin_on_maternal_disorder.relative_risk"
    )
    PAF_ATTRIBUTABLE_TO_HEMOGLOBIN: str = "risk_factor.hemoglobin_on_maternal_disorder.paf"

    @property
    def name(self):
        return "maternal_disorders"

    @property
    def log_name(self):
        return "maternal_disorders"


MATERNAL_DISORDERS = __MaternalDisorders()


class __MaternalHemorrhage(NamedTuple):
    RAW_INCIDENCE_RATE: str = "cause.maternal_hemorrhage.raw_incidence_rate"
    CSMR: str = "cause.maternal_hemorrhage.cause_specific_mortality_rate"
    INCIDENT_PROBABILITY: str = "cause.maternal_hemorrhage.incident_probability"

    RR_ATTRIBUTABLE_TO_HEMOGLOBIN: str = (
        "risk_factor.hemoglobin_on_maternal_hemorrhage.relative_risk"
    )
    PAF_ATTRIBUTABLE_TO_HEMOGLOBIN: str = "risk_factor.hemoglobin_on_maternal_hemorrhage.paf"

    MODERATE_HEMORRHAGE_PROBABILITY: str = (
        "covariate.moderate_hemorrhage_probability.estimate"
    )

    @property
    def name(self):
        return "maternal_hemorrhage"

    @property
    def log_name(self):
        return "maternal_hemorrhage"


MATERNAL_HEMORRHAGE = __MaternalHemorrhage()


class _Hemoglobin(NamedTuple):
    MEAN: TargetString = TargetString("risk_factor.hemoglobin.mean")
    STANDARD_DEVIATION: TargetString = TargetString(
        "risk_factor.hemoglobin.standard_deviation"
    )
    PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70: TargetString = TargetString(
        "risk_factor.hemoglobin.pregnant_proportion_below_70_gL"
    )

    @property
    def name(self):
        return "hemoglobin"

    @property
    def log_name(self):
        return "hemoglobin"


HEMOGLOBIN = _Hemoglobin()


class __MaternalBMI(NamedTuple):
    PREVALENCE_LOW_BMI_ANEMIC: str = "maternal_bmi.prevalance_low_bmi_anemic"
    PREVALENCE_LOW_BMI_NON_ANEMIC: str = "maternal_bmi.prevalence_low_bmi_non_anemic"

    @property
    def name(self):
        return "maternal_bmi"

    @property
    def log_name(self):
        return "Maternal BMI"


MATERNAL_BMI = __MaternalBMI()


class __MaternalInterventions(NamedTuple):
    IFA_COVERAGE: str = "maternal_interventions.ifa_coverage"
    IFA_EFFECT_SIZE: str = "maternal_interventions.ifa_effect_size"
    MMS_STILLBIRTH_RR: str = "maternal_interventions.mms_stillbirth_rr"
    BEP_STILLBIRTH_RR: str = "maternal_interventions.bep_stillbirth_rr"

    @property
    def name(self):
        return "maternal_interventions"

    @property
    def log_name(self):
        return "maternal interventions"


MATERNAL_INTERVENTIONS = __MaternalInterventions()

MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    PREGNANCY,
    LBWSG,
    MATERNAL_DISORDERS,
    MATERNAL_HEMORRHAGE,
    HEMOGLOBIN,
    MATERNAL_BMI,
    MATERNAL_INTERVENTIONS,
]
