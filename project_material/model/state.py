from enum import Enum

class DiseaseHealthState(Enum):
    SUSCEPTIBLE = 0
    INFECTIOUS = 1
    RECOVERED = 2
    DEAD = 3

class InfectiousSymptomState(Enum):
    NO_SYMPTOM = 0
    MILD_SYMPTOM = 1
    SEVERE_SYMPTOM = 2
    CRITICAL_SYMPTOM = 3

class TestResultState(Enum):
    TP = 0
    FP = 1
    TN = 3
    FN = 4

class UseHospitalBedState(Enum):
    NO = 0
    YES = 1

class UseICUBedState(Enum):
    NO = 0
    YES = 1

class UseVentilatorState(Enum):
    NO = 0
    YES = 1

class UseDrugXState(Enum):
    NO = 0
    YES = 1

class RecoveredImmunityState(Enum):
    WITH_IMMUNITY = 0
    WITHOUT_IMMUNITY = 1
    TBD = 2

class VaccineImmunityState(Enum):
    WITH_IMMUNITY = 0
    WITHOUT_IMMUNITY = 1
    TBD = 2

class RecoveredComplicationState(Enum):
    NO_COMPLICATION = 0
    MILD_COMPLICATION = 1
    SEVERE_COMPLICATION = 2

##############################
###### Helper functions ######
##############################

def number_disease_health_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.disease_health_state is state])

def number_infectious(model):
    return number_disease_health_state(model, DiseaseHealthState.INFECTIOUS)

def number_susceptible(model):
    return number_disease_health_state(model, DiseaseHealthState.SUSCEPTIBLE)

def number_recovered(model):
    return number_disease_health_state(model, DiseaseHealthState.RECOVERED)

def number_dead(model):
    return number_disease_health_state(model, DiseaseHealthState.DEAD)

def number_disease_health_state_test_confirmed(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if ((a.disease_health_state is state) & (
        a.test_result_on_disease_health_state is TestResultState.TP
    ))])

def number_infectious_test_confirmed(model):
    return number_disease_health_state_test_confirmed(model, DiseaseHealthState.INFECTIOUS)

def number_dead_test_confirmed(model):
    return number_disease_health_state_test_confirmed(model, DiseaseHealthState.DEAD)

def number_infectious_hospital_bed_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if ((a.infectious_hospital_bed_state is state) & (
        a.disease_health_state is not DiseaseHealthState.DEAD
    ))])

def number_infectious_using_hospital_bed(model):
    return number_infectious_hospital_bed_state(model, UseHospitalBedState.YES)

def number_infectious_icu_bed_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if ((a.infectious_icu_bed_state is state) & (
        a.disease_health_state is not DiseaseHealthState.DEAD
    ))])

def number_infectious_using_icu_bed(model):
    return number_infectious_icu_bed_state(model, UseICUBedState.YES)

def number_infectious_ventilator_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if ((a.infectious_ventilator_state is state) & (
        a.disease_health_state is not DiseaseHealthState.DEAD
    ))])

def number_infectious_using_ventilator(model):
    return number_infectious_ventilator_state(model, UseVentilatorState.YES)

def number_recovered_drugX_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if ((a.recovered_drugX_state is state) & (
        a.disease_health_state is not DiseaseHealthState.DEAD
    ))])

def number_recovered_using_drugX(model):
    return number_recovered_drugX_state(model, UseDrugXState.YES)

def number_infectious_symptom_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if ((a.infectious_symptom_state is state) & (
        a.disease_health_state is not DiseaseHealthState.DEAD
    ))])

def number_infectious_no_symptom(model):
    return number_infectious_symptom_state(model, InfectiousSymptomState.NO_SYMPTOM)

def number_infectious_mild_symptom(model):
    return number_infectious_symptom_state(model, InfectiousSymptomState.MILD_SYMPTOM)

def number_infectious_severe_symptom(model):
    return number_infectious_symptom_state(model, InfectiousSymptomState.SEVERE_SYMPTOM)

def number_infectious_critical_symptom(model):
    return number_infectious_symptom_state(model, InfectiousSymptomState.CRITICAL_SYMPTOM)

def number_test_done(model):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.new_test_done_over_current_time_unit is 1])

def number_test_result(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.test_result_on_disease_health_state is state])

def number_test_result_tp(model):
    return number_test_result(model, TestResultState.TP)

def number_test_result_fp(model):
    return number_test_result(model, TestResultState.FP)

def number_test_result_tn(model):
    return number_test_result(model, TestResultState.TN)

def number_test_result_fn(model):
    return number_test_result(model, TestResultState.FN)

def number_infectious_immunity_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if ((a.recovered_immunity_state is state) & (
        a.disease_health_state is not DiseaseHealthState.DEAD
    ))])

def number_recovered_with_immunity(model):
    return number_infectious_immunity_state(model, RecoveredImmunityState.WITH_IMMUNITY)

def number_recovered_without_immunity(model):
    return number_infectious_immunity_state(model, RecoveredImmunityState.WITHOUT_IMMUNITY)

def number_recovered_complication_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if ((a.recovered_complication_state is state) & (
        a.disease_health_state is not DiseaseHealthState.DEAD
    ))])

def number_recovered_no_complication(model):
    return number_recovered_complication_state(model, RecoveredComplicationState.NO_COMPLICATION)

def number_recovered_mild_complication(model):
    return number_recovered_complication_state(model, RecoveredComplicationState.MILD_COMPLICATION)

def number_recovered_severe_complication(model):
    return number_recovered_complication_state(model, RecoveredComplicationState.SEVERE_COMPLICATION)