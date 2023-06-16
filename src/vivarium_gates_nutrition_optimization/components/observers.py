from vivarium_public_health.metrics.disease import DiseaseObserver


class PregnancyObserver(DiseaseObserver):
    def __init__(self):
        super().__init__("pregnancy")
