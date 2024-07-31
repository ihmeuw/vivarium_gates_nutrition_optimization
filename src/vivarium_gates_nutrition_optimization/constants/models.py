from vivarium_gates_nutrition_optimization.constants import data_keys

###########################
# Pregnancy Model         #
###########################

PREGNANCY_MODEL_NAME = data_keys.PREGNANCY.name

# states
PREGNANT_STATE_NAME = "pregnant"
NOT_PREGNANT_STATE_NAME = "not_pregnant"
PARTURITION_STATE_NAME = "parturition"
POSTPARTUM_STATE_NAME = "postpartum"
PREGNANCY_MODEL_STATES = (
    NOT_PREGNANT_STATE_NAME,
    PREGNANT_STATE_NAME,
    PARTURITION_STATE_NAME,
    POSTPARTUM_STATE_NAME,
)

# outcomes
PARTIAL_TERM_OUTCOME = "partial_term"
LIVE_BIRTH_OUTCOME = "live_birth"
STILLBIRTH_OUTCOME = "stillbirth"
INVALID_OUTCOME = "invalid"  ## For sex of partial births
PREGNANCY_OUTCOMES = (
    PARTIAL_TERM_OUTCOME,
    LIVE_BIRTH_OUTCOME,
    STILLBIRTH_OUTCOME,
)

###########################
# Maternal Disorders      #
###########################

MATERNAL_DISORDERS_STATE_NAME = data_keys.MATERNAL_DISORDERS.name


###########################
# Maternal Hemorrhage     #
###########################

MATERNAL_HEMORRHAGE_STATE_NAME = data_keys.MATERNAL_HEMORRHAGE.name

INVALID_BMI_ANEMIA = "invalid"
LOW_BMI_ANEMIC = "low_bmi_anemic"
LOW_BMI_NON_ANEMIC = "low_bmi_non_anemic"
NORMAL_BMI_ANEMIC = "normal_bmi_anemic"
NORMAL_BMI_NON_ANEMIC = "normal_bmi_non_anemic"
BMI_ANEMIA_CATEGORIES = (
    INVALID_BMI_ANEMIA,
    LOW_BMI_ANEMIC,
    LOW_BMI_NON_ANEMIC,
    NORMAL_BMI_ANEMIC,
    NORMAL_BMI_NON_ANEMIC,
)

###########################
# Interventions           #
###########################

NO_TREATMENT = "uncovered"
IFA_SUPPLEMENTATION = "ifa"
MMS_SUPPLEMENTATION = "mms"
BEP_SUPPLEMENTATION = "bep"
SUPPLEMENTATION_CATEGORIES = (
    NO_TREATMENT,
    IFA_SUPPLEMENTATION,
    MMS_SUPPLEMENTATION,
    BEP_SUPPLEMENTATION,
)
