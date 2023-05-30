from pathlib import Path

import vivarium_gates_nutrition_optimization
from vivarium_gates_nutrition_optimization.constants import metadata

BASE_DIR = Path(vivarium_gates_nutrition_optimization.__file__).resolve().parent

ARTIFACT_ROOT = Path(f"/share/costeffectiveness/artifacts/{metadata.PROJECT_NAME}/")
MODEL_SPEC_DIR = BASE_DIR / "model_specifications"
RESULTS_ROOT = Path(f"/share/costeffectiveness/results/{metadata.PROJECT_NAME}/")
