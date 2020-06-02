from typing import List
import random
from mesa import Agent
from ..model.state import DiseaseHealthState, VaccineImmunityState, InfectiousSymptomState, TestResultState

class SocialDistancing(Agent):
    def __init__(self, unique_id, model, time_period: List[tuple], edge_threshold: List[float],
                 current_time, on_switch=False):
        assert len(time_period) == len(edge_threshold), \
            'ValueError: `time_period` and `edge_threshold` do not have the same length.'
        super().__init__(unique_id, model)
        self._list_slot_counter = None
        self.current_time = current_time
        self.edge_threshold = edge_threshold
        self.time_period = time_period # expects list of tuples, each tuple defines start and end time units
        self.on_switch = on_switch

    def check_timing(self):
        true_holder = False
        if self.on_switch:
            for index, time_period in enumerate(self.time_period):
                if self.current_time in range(time_period[0], time_period[1]+1):
                    true_holder = True
                    self._list_slot_counter = index
            return true_holder

    def assign_edge_threshold(self):
        return self.edge_threshold[self._list_slot_counter]

class Vaccine(Agent):
    def __init__(self, unique_id, model, agent, vaccine_success_rate: List[float],
                 time_period: List[tuple], prob_vaccinated: List[float], current_time, on_switch=False):
        assert len(time_period) == len(prob_vaccinated) == len(vaccine_success_rate), \
            'ValueError: `time_period`, `prob_vaccinated` and `vaccine_success_rate` do not have the same length.'
        super().__init__(unique_id, model)
        self._list_slot_counter = None
        self.agent = agent
        self.prob_vaccinated = prob_vaccinated
        self.vaccine_success_rate = vaccine_success_rate
        self.current_time = current_time
        self.time_period = time_period
        self.on_switch = on_switch

    def check_timing(self):
        true_holder = False
        if self.on_switch:
            for index, time_period in enumerate(self.time_period):
                if self.current_time in range(time_period[0], time_period[1]+1):
                    true_holder = True
                    self._list_slot_counter = index
            return true_holder

    def check_suitability(self):
        true_holder = False
        random_num = random.uniform(0, 1)
        if self.on_switch:
            if ((self.agent.disease_health_state is not DiseaseHealthState.DEAD) & (
                    self.agent.disease_health_state is not DiseaseHealthState.INFECTIOUS)):
                if self.agent.vaccine_immunity_state is not VaccineImmunityState.WITH_IMMUNITY:
                    if random_num < self.prob_vaccinated[self._list_slot_counter] * self.vaccine_success_rate[
                        self._list_slot_counter]:
                        true_holder = True
            return true_holder

    def assign_immune_state(self):
        self.agent.vaccine_immunity_state = VaccineImmunityState.WITH_IMMUNITY
        self.agent.time_units_when_successfully_gaining_immunity_from_vaccine.append(self.current_time)

class Testing(Agent):
    def __init__(self, unique_id, model, agent,
                    prob_tested_for_no_symptom: List[float],
                    prob_tested_for_mild_symptom: List[float],
                    prob_tested_for_severe_symptom: List[float],
                    prob_tested_for_critical_symptom: List[float],
                    test_sensitivity: List[float],
                    test_specificity: List[float],
                    time_period: List[tuple],
                    current_time=None, on_switch=True):
        assert len(time_period) == len(prob_tested_for_no_symptom) == len(prob_tested_for_mild_symptom) == \
               len(prob_tested_for_severe_symptom) == len(prob_tested_for_critical_symptom) == len(test_sensitivity) == \
               len(test_specificity), \
            'ValueError: `time_period`, `prob_tested_for_no_symptom`, `prob_tested_for_mild_symptom`, ' \
            '`prob_tested_for_severe_symptom`, `prob_tested_for_critical_symptom`, `test_sensitivity`, ' \
            'and `test_specificity` do not have the same length.'
        super().__init__(unique_id, model)
        self._list_slot_counter = None
        self._min_days_between_two_tests = 3 # Setting: At least x days between 2 adjacent tests
        self.agent = agent
        self.prob_tested_for_no_symptom = prob_tested_for_no_symptom
        self.prob_tested_for_mild_symptom = prob_tested_for_mild_symptom
        self.prob_tested_for_severe_symptom = prob_tested_for_severe_symptom
        self.prob_tested_for_critical_symptom = prob_tested_for_critical_symptom
        self.test_sensitivity = test_sensitivity
        self.test_specificity = test_specificity
        self.time_period = time_period
        self.current_time = current_time
        self.on_switch = on_switch

    def check_if_occurred_in_last_n_time_unit(self, occurrence: List[int], last_n_time_unit: int, current_time: int):
        '''Can be used to check if something occurs in the last n days.'''
        last_n_time_list = [t for t in range(current_time, current_time-last_n_time_unit, -1)]
        return any(i in occurrence for i in last_n_time_list)

    def check_timing(self):
        true_holder = False
        if self.on_switch:
            for index, time_period in enumerate(self.time_period):
                if self.current_time in range(time_period[0], time_period[1]+1):
                    true_holder = True
                    self._list_slot_counter = index
            return true_holder

    def check_suitability(self):
        true_holder = False
        random_num = random.uniform(0, 1)

        if self.on_switch:
            if ((self.agent.disease_health_state is DiseaseHealthState.SUSCEPTIBLE) |
                (self.agent.disease_health_state is DiseaseHealthState.RECOVERED) |
                ((self.agent.disease_health_state is DiseaseHealthState.INFECTIOUS) &
                (self.agent.infectious_symptom_state is InfectiousSymptomState.NO_SYMPTOM))):
                if random_num < self.prob_tested_for_no_symptom[self._list_slot_counter]:
                    if self.check_if_occurred_in_last_n_time_unit(
                            occurrence=self.agent.time_units_when_tested,
                            last_n_time_unit=self._min_days_between_two_tests,
                            current_time=self.current_time,
                    ) == False:
                        true_holder = True
            elif ((self.agent.disease_health_state is DiseaseHealthState.INFECTIOUS) &
                (self.agent.infectious_symptom_state is InfectiousSymptomState.MILD_SYMPTOM)):
                if random_num < self.prob_tested_for_mild_symptom[self._list_slot_counter]:
                    if self.check_if_occurred_in_last_n_time_unit(
                            occurrence=self.agent.time_units_when_tested,
                            last_n_time_unit=self._min_days_between_two_tests,
                            current_time=self.current_time,
                    ) == False:
                        true_holder = True
            elif ((self.agent.disease_health_state is DiseaseHealthState.INFECTIOUS) &
                (self.agent.infectious_symptom_state is InfectiousSymptomState.SEVERE_SYMPTOM)):
                if random_num < self.prob_tested_for_severe_symptom[self._list_slot_counter]:
                    if self.check_if_occurred_in_last_n_time_unit(
                            occurrence=self.agent.time_units_when_tested,
                            last_n_time_unit=self._min_days_between_two_tests,
                            current_time=self.current_time,
                    ) == False:
                        true_holder = True
            elif ((self.agent.disease_health_state is DiseaseHealthState.INFECTIOUS) &
                (self.agent.infectious_symptom_state is InfectiousSymptomState.CRITICAL_SYMPTOM)):
                if random_num < self.prob_tested_for_critical_symptom[self._list_slot_counter]:
                    if self.check_if_occurred_in_last_n_time_unit(
                            occurrence=self.agent.time_units_when_tested,
                            last_n_time_unit=self._min_days_between_two_tests,
                            current_time=self.current_time,
                    ) == False:
                        true_holder = True
            return true_holder

    def assign_test_result_if_applicable(self):
        random_num = random.uniform(0, 1)

        if self.check_timing():
            if self.check_suitability():
                self.agent.time_units_when_tested.append(self.current_time)
                self.agent.new_test_done_over_current_time_unit = 1
                self.agent.model.cumulative_test_done += 1

                if ((self.agent.disease_health_state is DiseaseHealthState.SUSCEPTIBLE) |
                    (self.agent.disease_health_state is DiseaseHealthState.RECOVERED)):
                    if random_num < self.test_specificity[self._list_slot_counter]:
                        self.agent.test_result_on_disease_health_state = TestResultState.TN
                    else:
                        self.agent.test_result_on_disease_health_state = TestResultState.FP

                elif (self.agent.disease_health_state is DiseaseHealthState.INFECTIOUS):
                    if random_num < self.test_sensitivity[self._list_slot_counter]:
                        self.agent.test_result_on_disease_health_state = TestResultState.TP
                    else:
                        self.agent.test_result_on_disease_health_state = TestResultState.FN