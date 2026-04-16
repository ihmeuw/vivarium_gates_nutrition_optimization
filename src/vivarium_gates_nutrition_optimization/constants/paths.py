from pathlib import Path

import vivarium_gates_nutrition_optimization
from vivarium_gates_nutrition_optimization.constants import metadata

BASE_DIR = Path(vivarium_gates_nutrition_optimization.__file__).resolve().parent

ARTIFACT_ROOT = Path(
    f"/mnt/team/simulation_science/pub/models/{metadata.PROJECT_NAME}/artifacts/"
)
MODEL_SPEC_DIR = BASE_DIR / "model_specifications"
RESULTS_ROOT = Path(f"/share/costeffectiveness/results/{metadata.PROJECT_NAME}/")
CSV_RAW_DATA_ROOT = BASE_DIR / "data" / "raw_data"

# Proportion of pregnant women with hemoglobin less than 70 g/L
PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70_CSV = (
    CSV_RAW_DATA_ROOT / "pregnant_proportion_with_hgb_below_70_age_specific.csv"
)

HEMOGLOBIN_PREGNANCY_ADJUSTMENT_FACTORS_CSV = (
    CSV_RAW_DATA_ROOT / "mean_pregnancy_adjustment_factor_draws.csv"
)

HEMOGLOBIN_MATERNAL_DISORDERS_PAF_CSV = (
    CSV_RAW_DATA_ROOT / "hemoglobin_and_maternal_disorders_pafs.csv"
)
PREVALENCE_LOW_BMI_ANEMIC_CSV = (
    CSV_RAW_DATA_ROOT / "prevalence_of_low_bmi_given_hemoglobin_below_10.csv"
)
PREVALENCE_LOW_BMI_NON_ANEMIC_CSV = (
    CSV_RAW_DATA_ROOT / "prevalence_of_low_bmi_given_hemoglobin_above_10.csv"
)
INTERVENTION_COVERAGE_BY_SCENARIO_CSV = CSV_RAW_DATA_ROOT / "coverage_by_scenario.csv"
