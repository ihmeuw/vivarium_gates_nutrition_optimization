from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_gates_nutrition_optimization.constants import data_keys, models

def Pregnancy():
    not_pregnant = SusceptibleState(models.PREGNANCY_MODEL_NAME)
    pregnant = DiseaseState(models.PREGNANCY_MODEL_NAME)
    pregnant.allow_self_transitions()
    return DiseaseModel(models.PREGNANCY_MODEL_NAME,states=[not_pregnant, pregnant])