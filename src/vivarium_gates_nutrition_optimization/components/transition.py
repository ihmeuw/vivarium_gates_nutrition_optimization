from vivarium_public_health.disease.transition import RateTransition
import pandas as pd

class PostpartumRateTransition(RateTransition):
    def compute_transition_rate(self, index):
        transition_rate = pd.Series(0, index=index)
        living = self.population_view.get(index, query='alive == "alive"').index
        base_rates = self.base_rate(living)
        joint_paf = self.joint_paf(living)
        transition_rate.loc[living] = base_rates * (1 - joint_paf)
        return transition_rate