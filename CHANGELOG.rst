**11.2 - 10/19/26**

 - Use the pytest-vivarium plugin for the shared test options and markers


**11.1 - 10/16/26**

 - Migrate artifact building off ``get_draws`` to vivarium_gbd_access 7 and vivarium_inputs 9: LBWSG birth exposure, hemoglobin exposure and relative risk, and the maternal disorders and anemia YLD rates now come through the shared functions, and the release-33 hemoglobin exposure standard deviation is read from the central comp draws under ``/mnt/team/simulation_science/pub/data/hemoglobin_release33``
 - Adopt the shared environment tooling; raise the floors to vivarium-engine 5.11.0 and vivarium-public-health 6.6.4
 - Remove the unfinished background morbidity feature and describe every pipeline producer and modifier
 - Fix a duplicated ``location`` index level in the population structure under pandas 2


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
