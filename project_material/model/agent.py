import sys
import logging
import random
from mesa import Agent
from ..model.state import DiseaseHealthState, RecoveredImmunityState, VaccineImmunityState, InfectiousSymptomState, \
    RecoveredComplicationState, UseHospitalBedState, UseICUBedState, UseVentilatorState, UseDrugXState, TestResultState
from ..helper.probability import age_generator, comorbidity_generator, probability_rescaler

logger = logging.getLogger('Logging for `agent.py`')
logger.setLevel(logging.WARNING) # Setting: Logging level

class HostAgent(Agent):
    def __init__(self, unique_id, model,
                    initial_disease_health_state,
                    initial_recovered_immunity_state,
                    prob_recovered_no_to_mild_complication,
                    prob_recovered_no_to_severe_complication,
                    prob_recovered_mild_to_no_complication,
                    prob_recovered_mild_to_severe_complication,
                    prob_recovered_severe_to_no_complication,
                    prob_recovered_severe_to_mild_complication,
                    prob_gain_immunity,
                    clinical_resource, social_distancing, vaccine, testing,
                 ):
        super().__init__(unique_id, model)
        self._stop_timer = None # Setting: If not `None`, simulation will stop at specified time
        self._current_timer = 0
        self._shuffle_behaviour_switch = True
        self._edge_weight_threshold_to_infect = 0.00 # the higher the harder to transmit virus; default at 0.00

        self.age = age_generator() # Setting: Currently using the AB age distribution from Census 2016
        self.sex = random.choice(['M', 'F']) # Setting: Simply assume probability to be M or F is 50:50
        self.comorbid_hypertension = comorbidity_generator('hypertension', self.age, self.sex)
        self.comorbid_diabetes = comorbidity_generator('diabetes', self.age, self.sex)
        self.comorbid_ihd = comorbidity_generator('ischemic heart disease', self.age, self.sex)
        self.comorbid_asthma = comorbidity_generator('asthma', self.age, self.sex)
        self.comorbid_cancer = comorbidity_generator('cancer', self.age, self.sex)

        self.disease_health_state = initial_disease_health_state
        self.initial_recovered_immunity_state = initial_recovered_immunity_state
        self.new_infection_tracker = {} # Track days and who an infectious host infect others
        self._timer_since_beginning_of_last_infection = None # Track number of days since the first day of the current/last infection
        self._timer_since_beginning_of_any_infection = 0  # Track number of days since the first day of any infection
        self._timer_since_beginning_of_last_onset_of_mild_symptom = None
        self._timer_since_beginning_of_last_onset_of_severe_or_critical_symptom = None

        self.test_result_on_disease_health_state = None
        self.new_test_done_over_current_time_unit = None
        self.infectious_symptom_state = None
        self.recovered_complication_state = None
        self.recovered_immunity_state = RecoveredImmunityState.WITHOUT_IMMUNITY
        self.vaccine_immunity_state = VaccineImmunityState.WITHOUT_IMMUNITY

        self.time_units_being_susceptible = []
        self.time_units_being_infectious = []
        self.time_units_being_recovered = []
        self.time_units_being_dead = []

        self.time_units_when_tested = []
        self.time_units_when_successfully_gaining_immunity_from_vaccine = []
        self.time_units_when_symptoms_are_severe_or_critical = []

        self.infectious_hospital_bed_state = None
        self.infectious_icu_bed_state = None
        self.infectious_ventilator_state = None
        self.recovered_drugX_state = None

        self.time_units_using_hospital_bed = []
        self.time_units_using_icu_bed = []
        self.time_units_using_ventilator = []
        self.time_units_using_drugX = []

        self.prob_spread_virus = None
        self.prob_recover = None
        self.prob_virus_kill_host = None
        self.prob_gain_immunity = prob_gain_immunity

        self.prob_infectious_no_symptom_maintained = None
        self.prob_infectious_no_to_mild_symptom = None
        self.prob_infectious_no_to_severe_symptom = None
        self.prob_infectious_no_to_critical_symptom = None

        self.prob_infectious_mild_symptom_maintained = None
        self.prob_infectious_mild_to_no_symptom = None
        self.prob_infectious_mild_to_severe_symptom = None
        self.prob_infectious_mild_to_critical_symptom = None

        self.prob_infectious_severe_symptom_maintained = None
        self.prob_infectious_severe_to_no_symptom = None
        self.prob_infectious_severe_to_mild_symptom = None
        self.prob_infectious_severe_to_critical_symptom = None

        self.prob_infectious_critical_symptom_maintained = None
        self.prob_infectious_critical_to_no_symptom = None
        self.prob_infectious_critical_to_mild_symptom = None
        self.prob_infectious_critical_to_severe_symptom = None

        self.prob_recovered_no_to_mild_complication = prob_recovered_no_to_mild_complication
        self.prob_recovered_no_to_severe_complication = prob_recovered_no_to_severe_complication
        self.prob_recovered_no_complication_maintained = 1 - (self.prob_recovered_no_to_mild_complication +
            self.prob_recovered_no_to_severe_complication)

        self.prob_recovered_mild_to_no_complication = prob_recovered_mild_to_no_complication
        self.prob_recovered_mild_to_severe_complication = prob_recovered_mild_to_severe_complication
        self.prob_recovered_mild_complication_maintained = 1 - (self.prob_recovered_mild_to_no_complication +
            self.prob_recovered_mild_to_severe_complication)

        self.prob_recovered_severe_to_no_complication = prob_recovered_severe_to_no_complication
        self.prob_recovered_severe_to_mild_complication = prob_recovered_severe_to_mild_complication
        self.prob_recovered_severe_complication_maintained = 1 - (self.prob_recovered_severe_to_no_complication +
            self.prob_recovered_severe_to_mild_complication)

        self.clinical_resource = clinical_resource
        self.social_distancing = social_distancing
        self.vaccine = vaccine
        self.testing = testing

    def try_social_distancing(self):
        self.social_distancing.current_time = self._current_timer
        if self.social_distancing.check_timing():
            self._edge_weight_threshold_to_infect = self.social_distancing.assign_edge_threshold()
        else:
            self._edge_weight_threshold_to_infect = 0.0

    def try_gain_immunity_from_vaccine(self):
        self.vaccine.agent = self
        self.vaccine.current_time = self._current_timer
        if self.vaccine.check_timing():
            if self.vaccine.check_suitability():
                self.vaccine.assign_immune_state()

    def try_infect_neighbors(self):
        if self.disease_health_state is DiseaseHealthState.INFECTIOUS:
            neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
            candidate_neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes) if
                                   (((agent.disease_health_state is DiseaseHealthState.SUSCEPTIBLE) and (
                                       agent.vaccine_immunity_state is not VaccineImmunityState.WITH_IMMUNITY
                                   )) or (
                                       (agent.disease_health_state is DiseaseHealthState.RECOVERED) and
                                       (agent.recovered_immunity_state is not RecoveredImmunityState.WITH_IMMUNITY) and
                                       (agent.vaccine_immunity_state is not VaccineImmunityState.WITH_IMMUNITY))
                                   )]
            newly_infected_neighbor_counter = 0

            for neighbor_agent in candidate_neighbors:
                if self.model.G[self.pos][neighbor_agent.pos]['weight'] > self._edge_weight_threshold_to_infect:
                    if self.random.random() < self.prob_spread_virus:
                        self.model.cumulative_infectious_cases += 1

                        newly_infected_neighbor_counter += 1
                        neighbor_agent.disease_health_state = DiseaseHealthState.INFECTIOUS
                        neighbor_agent._timer_since_beginning_of_last_infection = 0
                        neighbor_agent.infectious_symptom_state = InfectiousSymptomState.NO_SYMPTOM
                        neighbor_agent.recovered_complication_state = None

                if newly_infected_neighbor_counter >= 1:
                    self.new_infection_tracker.update({self._current_timer: newly_infected_neighbor_counter})
                    self.model.all_agents_new_infection_tracker.update({self.pos: self.new_infection_tracker})

    def try_recover_from_infection(self):
        prob_recover_with_no_complication = 0.70 # Setting: Assumed
        prob_recover_with_mild_complication = 0.20 # Setting: Assumed
        prob_recover_with_severe_complication = 0.10 # Setting: Assumed
        assert 1-(prob_recover_with_no_complication + prob_recover_with_mild_complication + \
                  prob_recover_with_severe_complication) <= 0.0000001, \
            'ValueError: `prob_recover_with_{}_complication` not sum to 1.00.'

        if self.disease_health_state is DiseaseHealthState.INFECTIOUS:
            if self.random.random() < self.prob_recover:
                self.disease_health_state = DiseaseHealthState.RECOVERED
                self.infectious_symptom_state = None
                self.recovered_complication_state = random.choices(
                    [RecoveredComplicationState.NO_COMPLICATION,
                     RecoveredComplicationState.MILD_COMPLICATION,
                     RecoveredComplicationState.SEVERE_COMPLICATION],
                    [prob_recover_with_no_complication,
                     prob_recover_with_mild_complication,
                     prob_recover_with_severe_complication], k=1
                )[0]
                self.try_gain_immunity_from_recovery()

    def try_gain_immunity_from_recovery(self):
        if self.random.random() < self.prob_gain_immunity:
            self.recovered_immunity_state = RecoveredImmunityState.WITH_IMMUNITY
            self.infectious_symptom_state = None

    def try_change_infectious_symptom_state(self):
        random_num = self.random.random()

        if self.disease_health_state is DiseaseHealthState.INFECTIOUS:
            if self.infectious_symptom_state is InfectiousSymptomState.NO_SYMPTOM:
                if (random_num >=0) & (random_num < self.prob_infectious_no_to_mild_symptom):
                    self.infectious_symptom_state = InfectiousSymptomState.MILD_SYMPTOM
                    self._timer_since_beginning_of_last_onset_of_mild_symptom = 0
                elif ((random_num >= self.prob_infectious_no_to_mild_symptom) and
                      (random_num < (self.prob_infectious_no_to_mild_symptom + self.prob_infectious_no_to_severe_symptom))):
                    self.infectious_symptom_state = InfectiousSymptomState.SEVERE_SYMPTOM
                    self._timer_since_beginning_of_last_onset_of_severe_or_critical_symptom = 0
                elif ((random_num >= (self.prob_infectious_no_to_mild_symptom + self.prob_infectious_no_to_severe_symptom)) and
                      (random_num < (self.prob_infectious_no_to_mild_symptom + self.prob_infectious_no_to_severe_symptom +
                                                self.prob_infectious_no_to_critical_symptom))):
                    self.infectious_symptom_state = InfectiousSymptomState.CRITICAL_SYMPTOM
                    self._timer_since_beginning_of_last_onset_of_severe_or_critical_symptom = 0

            elif self.infectious_symptom_state is InfectiousSymptomState.MILD_SYMPTOM:
                if (random_num >= 0) & (random_num < self.prob_infectious_mild_to_no_symptom):
                    self.infectious_symptom_state = InfectiousSymptomState.NO_SYMPTOM
                elif ((random_num >= self.prob_infectious_mild_to_no_symptom) and
                      (random_num < (self.prob_infectious_mild_to_no_symptom + self.prob_infectious_mild_to_severe_symptom))):
                    self.infectious_symptom_state = InfectiousSymptomState.SEVERE_SYMPTOM
                    self._timer_since_beginning_of_last_onset_of_severe_or_critical_symptom = 0
                elif ((random_num >= (self.prob_infectious_mild_to_no_symptom + self.prob_infectious_mild_to_severe_symptom)) and
                      (random_num < (self.prob_infectious_mild_to_no_symptom + self.prob_infectious_mild_to_severe_symptom +
                                                self.prob_infectious_mild_to_critical_symptom))):
                    self.infectious_symptom_state = InfectiousSymptomState.CRITICAL_SYMPTOM
                    self._timer_since_beginning_of_last_onset_of_severe_or_critical_symptom = 0

            elif self.infectious_symptom_state is InfectiousSymptomState.SEVERE_SYMPTOM:
                if (random_num >= 0) & (random_num < self.prob_infectious_severe_to_no_symptom):
                    self.infectious_symptom_state = InfectiousSymptomState.NO_SYMPTOM
                elif ((random_num >= self.prob_infectious_severe_to_no_symptom) and
                      (random_num < (
                              self.prob_infectious_severe_to_no_symptom + self.prob_infectious_severe_to_mild_symptom))):
                    self.infectious_symptom_state = InfectiousSymptomState.MILD_SYMPTOM
                    self._timer_since_beginning_of_last_onset_of_mild_symptom = 0
                elif ((random_num >= (self.prob_infectious_severe_to_no_symptom + self.prob_infectious_severe_to_mild_symptom)) and
                    (random_num < (self.prob_infectious_severe_to_no_symptom + self.prob_infectious_severe_to_mild_symptom +
                                              self.prob_infectious_mild_to_critical_symptom))):
                    self.infectious_symptom_state = InfectiousSymptomState.CRITICAL_SYMPTOM

            elif self.infectious_symptom_state is InfectiousSymptomState.CRITICAL_SYMPTOM:
                if (random_num >= 0) & (random_num < self.prob_infectious_critical_to_no_symptom):
                    self.infectious_symptom_state = InfectiousSymptomState.NO_SYMPTOM
                elif ((random_num >= self.prob_infectious_critical_to_no_symptom) and
                      (random_num < (
                              self.prob_infectious_critical_to_no_symptom + self.prob_infectious_critical_to_mild_symptom))):
                    self.infectious_symptom_state = InfectiousSymptomState.MILD_SYMPTOM
                    self._timer_since_beginning_of_last_onset_of_mild_symptom = 0
                elif ((random_num >= (self.prob_infectious_critical_to_no_symptom + self.prob_infectious_critical_to_mild_symptom)) and
                    (random_num < (self.prob_infectious_critical_to_no_symptom + self.prob_infectious_critical_to_mild_symptom +
                                              self.prob_infectious_critical_to_severe_symptom))):
                    self.infectious_symptom_state = InfectiousSymptomState.SEVERE_SYMPTOM

            else:
                raise Exception('`self.infectious_symptom_state` for the infectious host is missing')

    def try_change_recovered_complication_state(self):
        random_num = self.random.random()

        if self.disease_health_state is DiseaseHealthState.RECOVERED:
            if self.recovered_complication_state is RecoveredComplicationState.NO_COMPLICATION:
                if (random_num >=0) & (random_num < self.prob_recovered_no_to_mild_complication):
                    self.recovered_complication_state = RecoveredComplicationState.MILD_COMPLICATION
                elif ((random_num >= self.prob_recovered_no_to_mild_complication) and
                      (random_num < (
                              self.prob_recovered_no_to_mild_complication + self.prob_recovered_no_to_severe_complication))):
                    self.recovered_complication_state = RecoveredComplicationState.SEVERE_COMPLICATION

            elif self.recovered_complication_state is RecoveredComplicationState.MILD_COMPLICATION:
                if (random_num >= 0) & (random_num < self.prob_recovered_mild_to_no_complication):
                    self.recovered_complication_state = RecoveredComplicationState.NO_COMPLICATION
                elif ((random_num >= self.prob_recovered_mild_to_no_complication) and
                      (random_num < (
                              self.prob_recovered_mild_to_no_complication + self.prob_recovered_mild_to_severe_complication))):
                    self.recovered_complication_state = RecoveredComplicationState.SEVERE_COMPLICATION

            elif self.recovered_complication_state is RecoveredComplicationState.SEVERE_COMPLICATION:
                if (random_num >= 0) & (random_num < self.prob_recovered_severe_to_no_complication):
                    self.recovered_complication_state = RecoveredComplicationState.NO_COMPLICATION
                elif ((random_num >= self.prob_recovered_severe_to_no_complication) and
                      (random_num < (
                              self.prob_recovered_severe_to_no_complication + self.prob_recovered_severe_to_mild_complication))):
                    self.recovered_complication_state = RecoveredComplicationState.MILD_COMPLICATION
            else:
                raise Exception('`self.recovered_complication_state` for the recovered host is missing')

    def try_use_drugX(self):
        if (self.disease_health_state is DiseaseHealthState.RECOVERED) & (self.recovered_complication_state is
            RecoveredComplicationState.SEVERE_COMPLICATION):
            if self.clinical_resource.check_available_drugX() is True:
                if self.recovered_drugX_state is not UseDrugXState.YES:
                    self.model.cumulative_drugX_use_in_new_host_counts += 1
                self.recovered_drugX_state = UseDrugXState.YES
                self.clinical_resource.drugX_use_day_tracker += 1
                self.model.cumulative_drugX_use_in_days += 1
            else:
                self.recovered_drugX_state = UseDrugXState.NO

        if self.recovered_drugX_state is UseDrugXState.YES:
            if self.recovered_complication_state is not RecoveredComplicationState.SEVERE_COMPLICATION:
                self.recovered_drugX_state = UseDrugXState.NO
                self.clinical_resource.drugX_current_load -= 1

    def try_kill_host(self):
        if self.random.random() < self.prob_virus_kill_host:
            self.disease_health_state = DiseaseHealthState.DEAD
            self.model.cumulative_dead_cases += 1
            if self.test_result_on_disease_health_state is TestResultState.TP:
                self.model.cumulative_dead_test_confirmed_cases += 1

    def try_check_death(self):
        if self.disease_health_state is DiseaseHealthState.INFECTIOUS:
            self.try_kill_host()

    def try_test_disease_status(self):
        self.testing.agent = self
        self.testing.current_time = self._current_timer
        self.testing.assign_test_result_if_applicable()

        if self.pos not in self.model.all_agents_new_tested_as_true_positive:
            if self.test_result_on_disease_health_state is TestResultState.TP:
                if self.disease_health_state is DiseaseHealthState.INFECTIOUS:
                    self.model.cumulative_infectious_test_confirmed_cases += 1
                    self.model.all_agents_new_tested_as_true_positive.append(self.pos)

    def try_use_hospital_bed(self):
        if (self.infectious_symptom_state is InfectiousSymptomState.SEVERE_SYMPTOM) & (
                self.infectious_hospital_bed_state is not UseHospitalBedState.YES):
            if self.clinical_resource.check_available_hospital_bed() is True:
                self.infectious_hospital_bed_state = UseHospitalBedState.YES
                self.clinical_resource.hospital_bed_current_load += 1
                self.model.cumulative_hospital_bed_use_in_new_host_counts += 1
                if self.infectious_icu_bed_state is UseICUBedState.YES:
                    self.infectious_icu_bed_state = UseICUBedState.NO
                    self.clinical_resource.icu_bed_current_load -= 1

        if (self.infectious_hospital_bed_state is UseHospitalBedState.YES) & ((
                self.disease_health_state is not DiseaseHealthState.INFECTIOUS) | (
                self.infectious_symptom_state is not InfectiousSymptomState.SEVERE_SYMPTOM)):
            self.infectious_hospital_bed_state = UseHospitalBedState.NO
            self.clinical_resource.hospital_bed_current_load -= 1

        if self.infectious_hospital_bed_state is UseHospitalBedState.YES:
            self.clinical_resource.hospital_bed_use_day_tracker += 1
            self.model.cumulative_hospital_bed_use_in_days += 1

    def try_use_icu_bed(self):
        if (self.infectious_symptom_state is InfectiousSymptomState.CRITICAL_SYMPTOM) & (
                self.infectious_icu_bed_state is not UseICUBedState.YES):
            if self.clinical_resource.check_available_icu_bed() is True:
                self.infectious_icu_bed_state = UseICUBedState.YES
                self.clinical_resource.icu_bed_current_load += 1
                self.model.cumulative_icu_bed_use_in_new_host_counts += 1
                if self.infectious_hospital_bed_state is UseHospitalBedState.YES:
                    self.infectious_hospital_bed_state = UseHospitalBedState.NO
                    self.clinical_resource.hospital_bed_current_load -= 1

        if (self.infectious_icu_bed_state is UseICUBedState.YES) & ((
                self.disease_health_state is not DiseaseHealthState.INFECTIOUS) | (
                self.infectious_symptom_state is not InfectiousSymptomState.CRITICAL_SYMPTOM)):
            self.infectious_icu_bed_state = UseICUBedState.NO
            self.clinical_resource.icu_bed_current_load -= 1

        if self.infectious_icu_bed_state is UseICUBedState.YES:
            self.clinical_resource.icu_bed_use_day_tracker += 1
            self.model.cumulative_icu_bed_use_in_days += 1

    def try_use_ventilator(self):
        if (self.infectious_symptom_state in [InfectiousSymptomState.CRITICAL_SYMPTOM,
                InfectiousSymptomState.SEVERE_SYMPTOM]) & (self.infectious_ventilator_state
                is not UseVentilatorState.YES):
            if self.clinical_resource.check_available_ventilator() is True:
                self.infectious_ventilator_state = UseVentilatorState.YES
                self.clinical_resource.ventilator_current_load += 1
                self.model.cumulative_ventilator_use_in_new_host_counts += 1

        if (self.infectious_ventilator_state is UseVentilatorState.YES) & ((
                self.disease_health_state is not DiseaseHealthState.INFECTIOUS) | (
                self.infectious_symptom_state not in [InfectiousSymptomState.CRITICAL_SYMPTOM,
                InfectiousSymptomState.SEVERE_SYMPTOM])):
            self.infectious_ventilator_state = UseVentilatorState.NO
            self.clinical_resource.ventilator_current_load -= 1

        if self.infectious_ventilator_state is UseVentilatorState.YES:
            self.clinical_resource.ventilator_use_day_tracker += 1
            self.model.cumulative_ventilator_use_in_days += 1

    def track_time_unit_by_state(self):
        if self.disease_health_state is DiseaseHealthState.SUSCEPTIBLE:
            self.time_units_being_susceptible.append(self._current_timer)
        elif self.disease_health_state is DiseaseHealthState.INFECTIOUS:
            self.time_units_being_infectious.append(self._current_timer)
        elif self.disease_health_state is DiseaseHealthState.RECOVERED:
            self.time_units_being_recovered.append(self._current_timer)
        elif self.disease_health_state is DiseaseHealthState.DEAD:
            self.time_units_being_dead.append(self._current_timer)

        if self.infectious_hospital_bed_state is UseHospitalBedState.YES:
            self.time_units_using_hospital_bed.append(self._current_timer)

        if self.infectious_icu_bed_state is UseICUBedState.YES:
            self.time_units_using_icu_bed.append(self._current_timer)

        if self.infectious_ventilator_state is UseVentilatorState.YES:
            self.time_units_using_ventilator.append(self._current_timer)

        if self.recovered_drugX_state is UseDrugXState.YES:
            self.time_units_using_drugX.append(self._current_timer)

        if ((self.infectious_symptom_state is InfectiousSymptomState.SEVERE_SYMPTOM) |
                (self.infectious_symptom_state is InfectiousSymptomState.CRITICAL_SYMPTOM)):
            self.time_units_when_symptoms_are_severe_or_critical.append(self._current_timer)

    def construct_base_probability(self):
        if self._timer_since_beginning_of_last_infection:
            self.model.prob_spread_virus_dist.x = self._timer_since_beginning_of_last_infection
            self.model.prob_recover_dist.x = self._timer_since_beginning_of_last_infection
            self.model.prob_virus_kill_host_dist.x = self._timer_since_beginning_of_last_onset_of_severe_or_critical_symptom

            self.prob_spread_virus = self.model.prob_spread_virus_dist.get_pdf_prob_by_x()
            self.prob_recover = self.model.prob_recover_dist.get_pdf_prob_by_x()
            self.prob_virus_kill_host = self.model.prob_virus_kill_host_dist.get_cdf_prob_by_x()

            self.model.prob_infectious_no_to_mild_symptom_dist.x = self._timer_since_beginning_of_last_infection
            self.model.prob_infectious_no_to_severe_symptom_dist.x = self._timer_since_beginning_of_last_infection
            self.model.prob_infectious_no_to_critical_symptom_dist.x = self._timer_since_beginning_of_last_infection

            self.prob_infectious_no_to_mild_symptom = self.model.prob_infectious_no_to_mild_symptom_dist.get_pdf_prob_by_x()
            self.prob_infectious_no_to_severe_symptom = self.model.prob_infectious_no_to_severe_symptom_dist.get_pdf_prob_by_x()
            self.prob_infectious_no_to_critical_symptom = self.model.prob_infectious_no_to_critical_symptom_dist.get_pdf_prob_by_x()

            self.model.prob_infectious_mild_to_no_symptom_dist.x = self._timer_since_beginning_of_last_infection
            self.model.prob_infectious_mild_to_severe_symptom_dist.x = self._timer_since_beginning_of_last_infection
            self.model.prob_infectious_mild_to_critical_symptom_dist.x = self._timer_since_beginning_of_last_infection

            self.prob_infectious_mild_to_no_symptom = self.model.prob_infectious_mild_to_no_symptom_dist.get_pdf_prob_by_x()
            self.prob_infectious_mild_to_severe_symptom = self.model.prob_infectious_mild_to_severe_symptom_dist.get_pdf_prob_by_x()
            self.prob_infectious_mild_to_critical_symptom = self.model.prob_infectious_mild_to_critical_symptom_dist.get_pdf_prob_by_x()

            self.model.prob_infectious_severe_to_no_symptom_dist.x = self._timer_since_beginning_of_last_infection
            self.model.prob_infectious_severe_to_mild_symptom_dist.x = self._timer_since_beginning_of_last_infection
            self.model.prob_infectious_severe_to_critical_symptom_dist.x = self._timer_since_beginning_of_last_infection

            self.prob_infectious_severe_to_no_symptom = self.model.prob_infectious_severe_to_no_symptom_dist.get_pdf_prob_by_x()
            self.prob_infectious_severe_to_mild_symptom = self.model.prob_infectious_severe_to_mild_symptom_dist.get_pdf_prob_by_x()
            self.prob_infectious_severe_to_critical_symptom = self.model.prob_infectious_severe_to_critical_symptom_dist.get_pdf_prob_by_x()

            self.model.prob_infectious_critical_to_no_symptom_dist.x = self._timer_since_beginning_of_last_infection
            self.model.prob_infectious_critical_to_mild_symptom_dist.x = self._timer_since_beginning_of_last_infection
            self.model.prob_infectious_critical_to_severe_symptom_dist.x = self._timer_since_beginning_of_last_infection

            self.prob_infectious_critical_to_no_symptom = self.model.prob_infectious_critical_to_no_symptom_dist.get_pdf_prob_by_x()
            self.prob_infectious_critical_to_mild_symptom = self.model.prob_infectious_critical_to_mild_symptom_dist.get_pdf_prob_by_x()
            self.prob_infectious_critical_to_severe_symptom = self.model.prob_infectious_critical_to_severe_symptom_dist.get_pdf_prob_by_x()

        else:
            self.prob_spread_virus = 0
            self.prob_recover = 0
            self.prob_virus_kill_host = 0
            self.prob_infectious_no_to_mild_symptom = 0
            self.prob_infectious_no_to_severe_symptom = 0
            self.prob_infectious_no_to_critical_symptom = 0
            self.prob_infectious_mild_to_no_symptom = 0
            self.prob_infectious_mild_to_severe_symptom = 0
            self.prob_infectious_mild_to_critical_symptom = 0
            self.prob_infectious_severe_to_no_symptom = 0
            self.prob_infectious_severe_to_mild_symptom = 0
            self.prob_infectious_severe_to_critical_symptom = 0
            self.prob_infectious_critical_to_no_symptom = 0
            self.prob_infectious_critical_to_mild_symptom = 0
            self.prob_infectious_critical_to_severe_symptom = 0

    def update_probability_by_special_condition(self):
        modifier_from_hypertension = 0.05 # Setting: Assumed
        modifier_from_diabetes = 0.05  # Setting: Assumed
        modifier_from_ihd = 0.05  # Setting: Assumed
        modifier_from_asthma = 0.05  # Setting: Assumed
        modifier_from_cancer = 0.05  # Setting: Assumed
        modifier_from_old_age = 0.05 # Setting: Assumed
        modifier_from_severe_symptom = 0.05  # Setting: Assumed
        modifier_from_critical_symptom = 0.05  # Setting: Assumed
        modifier_from_critical_symptom_extra = 0.05/2  # Setting: Assumed
        modifier_from_absence_of_adequate_care = 0.05 # Setting: Assumed

        if self.comorbid_hypertension:
            self.prob_recover = self.prob_recover * (1-modifier_from_hypertension)
            self.prob_infectious_no_to_severe_symptom = self.prob_infectious_no_to_severe_symptom * (1+modifier_from_hypertension)
            self.prob_infectious_no_to_critical_symptom = self.prob_infectious_no_to_critical_symptom * (1+modifier_from_hypertension)

        if self.comorbid_diabetes:
            self.prob_recover = self.prob_recover * (1-modifier_from_diabetes)
            self.prob_infectious_no_to_severe_symptom = self.prob_infectious_no_to_severe_symptom * (1+modifier_from_diabetes)
            self.prob_infectious_no_to_critical_symptom = self.prob_infectious_no_to_critical_symptom * (1+modifier_from_diabetes)

        if self.comorbid_ihd:
            self.prob_recover = self.prob_recover * (1-modifier_from_ihd)
            self.prob_infectious_no_to_severe_symptom = self.prob_infectious_no_to_severe_symptom * (1+modifier_from_ihd)
            self.prob_infectious_no_to_critical_symptom = self.prob_infectious_no_to_critical_symptom * (1+modifier_from_ihd)

        if self.comorbid_asthma:
            self.prob_recover = self.prob_recover * (1-modifier_from_asthma)
            self.prob_infectious_no_to_severe_symptom = self.prob_infectious_no_to_severe_symptom * (1+modifier_from_asthma)
            self.prob_infectious_no_to_critical_symptom = self.prob_infectious_no_to_critical_symptom * (1+modifier_from_asthma)

        if self.comorbid_cancer:
            self.prob_recover = self.prob_recover * (1-modifier_from_cancer)
            self.prob_infectious_no_to_severe_symptom = self.prob_infectious_no_to_severe_symptom * (1+modifier_from_cancer)
            self.prob_infectious_no_to_critical_symptom = self.prob_infectious_no_to_critical_symptom * (1+modifier_from_cancer)

        if self.age >= 60:
            self.prob_recover = self.prob_recover * (1-modifier_from_old_age)
            self.prob_infectious_no_to_severe_symptom = self.prob_infectious_no_to_severe_symptom * (1+modifier_from_old_age)
            self.prob_infectious_no_to_critical_symptom = self.prob_infectious_no_to_critical_symptom * (1+modifier_from_old_age)

        if self.infectious_symptom_state is InfectiousSymptomState.SEVERE_SYMPTOM:
            self.prob_recover = self.prob_recover * (1-modifier_from_severe_symptom)
            self.prob_virus_kill_host = self.prob_virus_kill_host * (1+modifier_from_severe_symptom)

            if self.infectious_hospital_bed_state is UseHospitalBedState.YES:
                self.prob_recover = self.prob_recover * (1+modifier_from_absence_of_adequate_care)
                self.prob_infectious_severe_to_critical_symptom = self.prob_infectious_severe_to_critical_symptom * (
                            1-modifier_from_absence_of_adequate_care)
                self.prob_virus_kill_host = self.prob_virus_kill_host * (
                            1-modifier_from_absence_of_adequate_care)

            else:
                self.prob_recover = self.prob_recover * (1-modifier_from_absence_of_adequate_care)
                self.prob_infectious_severe_to_critical_symptom = self.prob_infectious_severe_to_critical_symptom * (
                            1+modifier_from_absence_of_adequate_care)
                self.prob_virus_kill_host = self.prob_virus_kill_host * (
                            1+modifier_from_absence_of_adequate_care)

            if self.infectious_ventilator_state is UseVentilatorState.YES:
                self.prob_recover = self.prob_recover * (1-modifier_from_absence_of_adequate_care)
                self.prob_infectious_severe_to_critical_symptom = self.prob_infectious_severe_to_critical_symptom * (
                            1-modifier_from_absence_of_adequate_care)
                self.prob_virus_kill_host = self.prob_virus_kill_host * (
                            1-modifier_from_absence_of_adequate_care)

            else:
                self.prob_recover = self.prob_recover * (1-modifier_from_absence_of_adequate_care)
                self.prob_infectious_severe_to_critical_symptom = self.prob_infectious_severe_to_critical_symptom * (
                            1+modifier_from_absence_of_adequate_care)
                self.prob_virus_kill_host = self.prob_virus_kill_host * (
                            1+modifier_from_absence_of_adequate_care)

        if self.infectious_symptom_state is InfectiousSymptomState.CRITICAL_SYMPTOM:
            self.prob_recover = self.prob_recover * (1-modifier_from_critical_symptom)
            self.prob_virus_kill_host = self.prob_virus_kill_host * (
                    1+modifier_from_critical_symptom+modifier_from_critical_symptom_extra)

            if self.infectious_icu_bed_state is UseICUBedState.YES:
                self.prob_recover = self.prob_recover * (1+modifier_from_absence_of_adequate_care)
                self.prob_infectious_critical_to_severe_symptom = self.prob_infectious_critical_to_severe_symptom * (
                            1+modifier_from_absence_of_adequate_care)
                self.prob_virus_kill_host = self.prob_virus_kill_host * (
                            1-modifier_from_absence_of_adequate_care)

            else:
                self.prob_recover = self.prob_recover * (1-modifier_from_absence_of_adequate_care)
                self.prob_infectious_critical_to_severe_symptom = self.prob_infectious_critical_to_severe_symptom * (
                            1-modifier_from_absence_of_adequate_care)
                self.prob_virus_kill_host = self.prob_virus_kill_host * (
                            1+modifier_from_absence_of_adequate_care)

            if self.infectious_ventilator_state is UseVentilatorState.YES:
                self.prob_recover = self.prob_recover * (1+modifier_from_absence_of_adequate_care)
                self.prob_infectious_critical_to_severe_symptom = self.prob_infectious_critical_to_severe_symptom * (
                            1+modifier_from_absence_of_adequate_care)
                self.prob_virus_kill_host = self.prob_virus_kill_host * (
                            1-modifier_from_absence_of_adequate_care)

            else:
                self.prob_recover = self.prob_recover * (1-modifier_from_absence_of_adequate_care)
                self.prob_infectious_critical_to_severe_symptom = self.prob_infectious_critical_to_severe_symptom * (
                            1-modifier_from_absence_of_adequate_care)
                self.prob_virus_kill_host = self.prob_virus_kill_host * (
                            1+modifier_from_absence_of_adequate_care)

    def final_probability_update(self):
        self.prob_infectious_no_symptom_maintained = 1 - (self.prob_infectious_no_to_mild_symptom +
            self.prob_infectious_no_to_severe_symptom + self.prob_infectious_no_to_critical_symptom)

        self.prob_infectious_mild_symptom_maintained = 1 - (self.prob_infectious_mild_to_no_symptom +
            self.prob_infectious_mild_to_severe_symptom + self.prob_infectious_mild_to_critical_symptom)

        self.prob_infectious_severe_symptom_maintained = 1 - (self.prob_infectious_severe_to_no_symptom +
            self.prob_infectious_severe_to_mild_symptom + self.prob_infectious_severe_to_critical_symptom)

        self.prob_infectious_critical_symptom_maintained = 1 - (self.prob_infectious_critical_to_no_symptom +
            self.prob_infectious_critical_to_mild_symptom + self.prob_infectious_critical_to_severe_symptom)

        # Rescale if `prob_infectious_{}_symptom_maintained` is less than 0
        if self.prob_infectious_no_symptom_maintained < 0:
            logger.warning('WARNING:`prob_infectious_no_symptom_maintained` for agent {} is less than 0, rescaling applied.'.format(
                    self.pos))
            self.prob_infectious_no_symptom_maintained = 0
            self.prob_infectious_no_to_mild_symptom, self.prob_infectious_no_to_severe_symptom, \
            self.prob_infectious_no_to_critical_symptom = \
                probability_rescaler(self.prob_infectious_no_to_mild_symptom,
                                     self.prob_infectious_no_to_severe_symptom,
                                     self.prob_infectious_no_to_critical_symptom)

        if self.prob_infectious_mild_symptom_maintained < 0:
            logger.warning('WARNING:`prob_infectious_mild_symptom_maintained` for agent {} is less than 0, rescaling applied.'.format(
                    self.pos))
            self.prob_infectious_mild_symptom_maintained = 0
            self.prob_infectious_mild_to_no_symptom, self.prob_infectious_mild_to_severe_symptom, \
            self.prob_infectious_mild_to_critical_symptom = \
                probability_rescaler(self.prob_infectious_mild_to_no_symptom,
                                     self.prob_infectious_mild_to_severe_symptom,
                                     self.prob_infectious_mild_to_critical_symptom)

        if self.prob_infectious_severe_symptom_maintained < 0:
            logger.warning('WARNING:`prob_infectious_severe_symptom_maintained` for agent {} is less than 0, rescaling applied.'.format(
                    self.pos))
            self.prob_infectious_severe_symptom_maintained = 0
            self.prob_infectious_severe_to_no_symptom, self.prob_infectious_severe_to_mild_symptom, \
            self.prob_infectious_severe_to_critical_symptom = \
                probability_rescaler(self.prob_infectious_severe_to_no_symptom,
                                     self.prob_infectious_severe_to_mild_symptom,
                                     self.prob_infectious_severe_to_critical_symptom)

        if self.prob_infectious_critical_symptom_maintained < 0:
            logger.warning('WARNING:`prob_infectious_critical_symptom_maintained` for agent {} is less than 0, rescaling applied.'.format(
                    self.pos))
            self.prob_infectious_critical_symptom_maintained = 0
            self.prob_infectious_critical_to_no_symptom, self.prob_infectious_critical_to_mild_symptom, \
            self.prob_infectious_critical_to_severe_symptom = \
                probability_rescaler(self.prob_infectious_critical_to_no_symptom,
                                     self.prob_infectious_critical_to_mild_symptom,
                                     self.prob_infectious_critical_to_severe_symptom)

    def validate_probability_setting(self):
        assert self.prob_infectious_no_symptom_maintained >= 0, 'ValueError: `prob_infectious_no_symptom_maintained`' \
                                                                'is less than 0.'
        assert self.prob_infectious_mild_symptom_maintained >= 0, 'ValueError: `prob_infectious_mild_symptom_maintained`' \
                                                                'is less than 0.'
        assert self.prob_infectious_severe_symptom_maintained >= 0, 'ValueError: `prob_infectious_severe_symptom_maintained`' \
                                                                'is less than 0.'
        assert self.prob_infectious_critical_symptom_maintained >= 0, 'ValueError: `prob_infectious_critical_symptom_maintained`' \
                                                                'is less than 0.'
        assert self.prob_recovered_no_complication_maintained >= 0, 'ValueError: `prob_recovered_no_complication_maintained`' \
                                                                'is less than 0.'
        assert self.prob_recovered_mild_complication_maintained >= 0, 'ValueError: `prob_recovered_mild_complication_maintained`' \
                                                                'is less than 0.'
        assert self.prob_recovered_severe_complication_maintained >= 0, 'ValueError: `prob_recovered_severe_complication_maintained`' \
                                                                'is less than 0.'

    def update_time_variable(self):
        if self.disease_health_state is not DiseaseHealthState.SUSCEPTIBLE:
            self._timer_since_beginning_of_last_infection += 1
            self._timer_since_beginning_of_any_infection += 1

            if ((self.infectious_symptom_state is InfectiousSymptomState.SEVERE_SYMPTOM) |
                    (self.infectious_symptom_state is InfectiousSymptomState.CRITICAL_SYMPTOM)):
                self._timer_since_beginning_of_last_onset_of_severe_or_critical_symptom += 1

            if self.infectious_symptom_state is InfectiousSymptomState.MILD_SYMPTOM:
                self._timer_since_beginning_of_last_onset_of_mild_symptom += 1

        else:
            self._timer_since_beginning_of_last_infection = None

    def initial_variable_reset(self):
        self.new_test_done_over_current_time_unit = None

    def end_variable_reset(self):
        pass

    def step(self):
        self._current_timer += 1

        if self._stop_timer:
            if (self._current_timer > self._stop_timer):
                logger.info('INFO: Execution ended when the `_unit_timer_stopper` has reached the specified time.')
                sys.exit()

        initial_function_list = [
            self.initial_variable_reset,
            self.construct_base_probability,
            self.update_probability_by_special_condition,
            self.final_probability_update,
            self.validate_probability_setting,
            self.try_social_distancing,
            self.try_test_disease_status,
        ]

        function_list = [
            self.try_infect_neighbors,
            self.try_recover_from_infection,
            self.try_check_death,
            self.try_change_infectious_symptom_state,
            self.try_change_recovered_complication_state,
            self.try_use_drugX,
            self.try_use_hospital_bed,
            self.try_use_icu_bed,
            self.try_use_ventilator,
            self.try_gain_immunity_from_vaccine,
        ]

        end_function_list = [
            self.track_time_unit_by_state,
            self.update_time_variable,
            self.end_variable_reset,
        ]

        if self._shuffle_behaviour_switch:
            random.shuffle(function_list)

        full_function_list = initial_function_list + function_list + end_function_list
        [f() for f in full_function_list]

    def advance(self):
        self.step()

    ### Class helper functions ###
    def describe_agent_profile(self, show_prob_infectious_symptom_state_change=False):
        print('/////////////////////////////')
        print('Agent {} in day {}'.format(self.pos, self._current_timer))

        if show_prob_infectious_symptom_state_change:
            print('Agent health disease state:', self.disease_health_state)
            print('Agent infectious symptom state:', self.infectious_symptom_state)
            print('Agent prob. remaining no symptom:', self.prob_infectious_no_symptom_maintained)
            print('Agent prob. no to mild symptom:', self.prob_infectious_no_to_mild_symptom)
            print('Agent prob. no to severe symptom:', self.prob_infectious_no_to_severe_symptom)
            print('Agent prob. no to critical symptom:', self.prob_infectious_no_to_critical_symptom)
            print('Agent prob. remaining mild symptom:', self.prob_infectious_mild_symptom_maintained)
            print('Agent prob. mild to no symptom:', self.prob_infectious_mild_to_no_symptom)
            print('Agent prob. mild to severe symptom:', self.prob_infectious_mild_to_severe_symptom)
            print('Agent prob. mild to critical symptom:', self.prob_infectious_mild_to_critical_symptom)
            print('Agent prob. remaining severe symptom:', self.prob_infectious_severe_symptom_maintained)
            print('Agent prob. severe to no symptom:', self.prob_infectious_severe_to_no_symptom)
            print('Agent prob. severe to mild symptom:', self.prob_infectious_severe_to_mild_symptom)
            print('Agent prob. severe to critical symptom:', self.prob_infectious_severe_to_critical_symptom)
            print('Agent prob. remaining critical symptom:', self.prob_infectious_critical_symptom_maintained)
            print('Agent prob. critical to no symptom:', self.prob_infectious_critical_to_no_symptom)
            print('Agent prob. critical to mild symptom:', self.prob_infectious_critical_to_mild_symptom)
            print('Agent prob. critical to severe symptom:', self.prob_infectious_critical_to_severe_symptom)

        print('/////////////////////////////')
        print('/////////////////////////////'+'\n')
