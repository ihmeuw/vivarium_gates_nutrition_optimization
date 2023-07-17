from vivarium_public_health.disease.transition import RateTransition
import pandas as pd

class ParturitionSelectionRateTransition(RateTransition):
    def compute_transition_rate(self, index):
        transition_rate = pd.Series(0, index=index)
        sub_pop = self.population_view.get(index, query="(alive == 'alive') & (pregnancy == 'parturition')").index
        base_rates = self.base_rate(sub_pop)
        joint_paf = self.joint_paf(sub_pop)
        transition_rate.loc[sub_pop] = base_rates * (1 - joint_paf)
        return transition_rate