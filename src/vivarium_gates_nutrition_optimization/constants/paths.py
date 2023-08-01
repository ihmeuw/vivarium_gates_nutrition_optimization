from pathlib import Path

import vivarium_gates_nutrition_optimization
from vivarium_gates_nutrition_optimization.constants import metadata

BASE_DIR = Path(vivarium_gates_nutrition_optimization.__file__).resolve().parent

ARTIFACT_ROOT = Path(f"/share/costeffectiveness/artifacts/{metadata.PROJECT_NAME}/")
MODEL_SPEC_DIR = BASE_DIR / "model_specifications"
RESULTS_ROOT = Path(f"/share/costeffectiveness/results/{metadata.PROJECT_NAME}/")
CSV_RAW_DATA_ROOT = BASE_DIR / "data" / "raw_data"

# Proportion of pregnant women with hemoglobin less than 70 g/L
PREGNANT_PROPORTION_WITH_HEMOGLOBIN_BELOW_70_CSV = (
    CSV_RAW_DATA_ROOT / "pregnant_proportion_with_hgb_below_70_age_specific.csv"
)
HEMOGLOBIN_MATERNAL_DISORDERS_PAF_CSV = (
    CSV_RAW_DATA_ROOT / "hemoglobin_and_maternal_disorders_pafs.csv"
)
PREVALENCE_LOW_BMI_ANEMIC_CSV = (
    CSV_RAW_DATA_ROOT / 'prevalence_of_low_bmi_given_hemoglobin_below_10.csv'
)
PREVALENCE_LOW_BMI_NON_ANEMIC_CSV = (
    CSV_RAW_DATA_ROOT / 'prevalence_of_low_bmi_given_hemoglobin_above_10.csv'
)
MATERNAL_INTERVENTION_COVERAGE_CSV = (
    CSV_RAW_DATA_ROOT / 'simulation_intervention_coverage.csv'
)

