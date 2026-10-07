**11.1 - 10/05/26**

 - Raise the floors to vivarium-engine>=5.11.0 and vivarium-public-health>=6.6.4
 - Add a description to every value pipeline producer and modifier registration
 - Remove background morbidity: the commented-out BackgroundMorbidity component and ParturitionExclusionState, the BACKGROUND_MORBIDITY data key, and the loader and extra_gbd functions that only it used. The feature was switched off in 11.0 and never finished; the design remains in the git history
 - Include the test extra in the data extra
 - Remove the commented-out ParturitionExclusionState from components/disease.py
 - Raise the data extra pins to vivarium_inputs>=9.0.0,<10.0.0 and vivarium_gbd_access>=7.0.0,<8.0.0
 - Source the hemoglobin-below-70 proportion through the ``impairment-cause`` machinery entity and the hemoglobin mean exposure through measures.get_exposure, both still under release 33; the release-33 exposure standard deviation has no best model version and raises until it is sourced
 - Migrate the live get_draws calls in data/extra_gbd.py to gbd_access 7: LBWSG birth exposure through measures.get_birth_exposure, maternal disorders and anemia sequelae YLD rates through base_data.get_machinery_estimates, and the hemoglobin maternal disorders relative risk through measures.get_relative_risk
 - Replace utility_data.get_location_id with utility_data.resolve_location and the model-local load_standard_data with vivarium_inputs.interface.load_standard_data

**11.0 - 3/26/24**

 - Update to 2021 data for Ethiopia, bug-fix changes to pregnancy observation, changes to stratification (anemia by pregnancy state and pregnancy transitions by pregnancy outcomes), temporary removal of the IFA scenario, and removal of background morbidity.

**1.0.3 - 3/12/24**

 - Update Mortality observer to incorporate update in VPH Mortality Observer

**1.0.2 - 03/11/24**

 - Remove universal BEP scenario and targeted BEP/no MMS scenario and include universal IFA scenario and targeted BEP/IFA scenario

**1.0.1 - 10/02/23**

 - Refactor to use vivarium Components

**1.0.0 - 09/18/23**

 - Release to 1.0.0 with Wave 1 production runs
