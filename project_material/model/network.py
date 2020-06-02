import itertools
import random
import math
import networkx as nx
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

from ..model.state import DiseaseHealthState, InfectiousSymptomState, RecoveredImmunityState, number_susceptible, number_infectious, \
    number_recovered, number_disease_health_state, number_dead, number_infectious_no_symptom, number_infectious_mild_symptom, \
    number_infectious_severe_symptom, number_infectious_critical_symptom, number_recovered_no_complication, number_recovered_mild_complication, \
    number_recovered_severe_complication, number_infectious_using_hospital_bed, number_infectious_using_icu_bed, \
    number_infectious_using_ventilator, number_recovered_using_drugX, number_infectious_test_confirmed, number_test_done, \
    number_dead_test_confirmed, number_disease_health_state_test_confirmed
from ..helper.generic import mean_r0, return_time, return_total_n, cumulative_total_infectious, cumulative_total_dead, \
    cumulative_total_test_done, rate_cumulative_infectious, rate_cumulative_dead, rate_cumulative_infectious_test_confirmed, \
    rate_cumulative_dead_test_confirmed, rate_cumulative_test_done, cumulative_total_infectious_test_confirmed, \
    cumulative_total_dead_test_confirmed
from ..model.agent import HostAgent
from ..model.clinical_resource import ClinicalResource
from ..model.intervention import SocialDistancing, Vaccine, Testing
from ..helper.time_distribution import GammaProbabilityGenerator

class HostNetwork(Model):
    # id generator to track run number in batch run data
    id_gen = itertools.count(1)
    rate_denominator = 1000000

    def __init__(self, num_nodes, avg_node_degree, initial_outbreak_size,

                    prob_spread_virus_gamma_shape,
                    prob_spread_virus_gamma_scale,
                    prob_spread_virus_gamma_loc,
                    prob_spread_virus_gamma_magnitude_multiplier,

                    prob_recover_gamma_shape,
                    prob_recover_gamma_scale,
                    prob_recover_gamma_loc,
                    prob_recover_gamma_magnitude_multiplier,

                    prob_virus_kill_host_gamma_shape,
                    prob_virus_kill_host_gamma_scale,
                    prob_virus_kill_host_gamma_loc,
                    prob_virus_kill_host_gamma_magnitude_multiplier,

                    prob_infectious_no_to_mild_symptom_gamma_shape,
                    prob_infectious_no_to_mild_symptom_gamma_scale,
                    prob_infectious_no_to_mild_symptom_gamma_loc,
                    prob_infectious_no_to_mild_symptom_gamma_magnitude_multiplier,

                    prob_infectious_no_to_severe_symptom_gamma_shape,
                    prob_infectious_no_to_severe_symptom_gamma_scale,
                    prob_infectious_no_to_severe_symptom_gamma_loc,
                    prob_infectious_no_to_severe_symptom_gamma_magnitude_multiplier,

                    prob_infectious_no_to_critical_symptom_gamma_shape,
                    prob_infectious_no_to_critical_symptom_gamma_scale,
                    prob_infectious_no_to_critical_symptom_gamma_loc,
                    prob_infectious_no_to_critical_symptom_gamma_magnitude_multiplier,

                    prob_infectious_mild_to_no_symptom_gamma_shape,
                    prob_infectious_mild_to_no_symptom_gamma_scale,
                    prob_infectious_mild_to_no_symptom_gamma_loc,
                    prob_infectious_mild_to_no_symptom_gamma_magnitude_multiplier,

                    prob_infectious_mild_to_severe_symptom_gamma_shape,
                    prob_infectious_mild_to_severe_symptom_gamma_scale,
                    prob_infectious_mild_to_severe_symptom_gamma_loc,
                    prob_infectious_mild_to_severe_symptom_gamma_magnitude_multiplier,

                    prob_infectious_mild_to_critical_symptom_gamma_shape,
                    prob_infectious_mild_to_critical_symptom_gamma_scale,
                    prob_infectious_mild_to_critical_symptom_gamma_loc,
                    prob_infectious_mild_to_critical_symptom_gamma_magnitude_multiplier,

                    prob_infectious_severe_to_no_symptom_gamma_shape,
                    prob_infectious_severe_to_no_symptom_gamma_scale,
                    prob_infectious_severe_to_no_symptom_gamma_loc,
                    prob_infectious_severe_to_no_symptom_gamma_magnitude_multiplier,

                    prob_infectious_severe_to_mild_symptom_gamma_shape,
                    prob_infectious_severe_to_mild_symptom_gamma_scale,
                    prob_infectious_severe_to_mild_symptom_gamma_loc,
                    prob_infectious_severe_to_mild_symptom_gamma_magnitude_multiplier,

                    prob_infectious_severe_to_critical_symptom_gamma_shape,
                    prob_infectious_severe_to_critical_symptom_gamma_scale,
                    prob_infectious_severe_to_critical_symptom_gamma_loc,
                    prob_infectious_severe_to_critical_symptom_gamma_magnitude_multiplier,

                    prob_infectious_critical_to_no_symptom_gamma_shape,
                    prob_infectious_critical_to_no_symptom_gamma_scale,
                    prob_infectious_critical_to_no_symptom_gamma_loc,
                    prob_infectious_critical_to_no_symptom_gamma_magnitude_multiplier,

                    prob_infectious_critical_to_mild_symptom_gamma_shape,
                    prob_infectious_critical_to_mild_symptom_gamma_scale,
                    prob_infectious_critical_to_mild_symptom_gamma_loc,
                    prob_infectious_critical_to_mild_symptom_gamma_magnitude_multiplier,

                    prob_infectious_critical_to_severe_symptom_gamma_shape,
                    prob_infectious_critical_to_severe_symptom_gamma_scale,
                    prob_infectious_critical_to_severe_symptom_gamma_loc,
                    prob_infectious_critical_to_severe_symptom_gamma_magnitude_multiplier,

                    prob_recovered_no_to_mild_complication,
                    prob_recovered_no_to_severe_complication,
                    prob_recovered_mild_to_no_complication,
                    prob_recovered_mild_to_severe_complication,
                    prob_recovered_severe_to_no_complication,
                    prob_recovered_severe_to_mild_complication,
                    prob_gain_immunity,

                    hospital_bed_capacity_as_percent_of_population,
                    hospital_bed_cost_per_day,

                    icu_bed_capacity_as_percent_of_population,
                    icu_bed_cost_per_day,

                    ventilator_capacity_as_percent_of_population,
                    ventilator_cost_per_day,

                    drugX_capacity_as_percent_of_population,
                    drugX_cost_per_day,
                 ):

        self.uid = next(self.id_gen)
        self.set_network_seed = 888 # Setting: Accurately set to None or specific seed
        self.set_initial_infectious_node_seed = 888 # Setting: Accurately set to None or specific seed

        self._current_timer = 0
        self._last_n_time_unit_for_mean_r0 = 10 # SETTING: Smoothing mean R0
        self.num_nodes = num_nodes
        self.avg_node_degree = avg_node_degree
        prob = self.avg_node_degree / self.num_nodes
        self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=prob, seed=self.set_network_seed)
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.initial_outbreak_size = initial_outbreak_size if initial_outbreak_size <= num_nodes else num_nodes
        self.all_agents_new_infection_tracker = {}
        self.all_agents_new_tested_as_true_positive = []

        self.cumulative_infectious_cases = self.initial_outbreak_size
        self.cumulative_dead_cases = 0
        self.cumulative_test_done = 0
        self.cumulative_infectious_test_confirmed_cases = 0
        self.cumulative_dead_test_confirmed_cases = 0

        self.cumulative_hospital_bed_use_in_new_host_counts = 0
        self.cumulative_icu_bed_use_in_new_host_counts = 0
        self.cumulative_ventilator_use_in_new_host_counts = 0
        self.cumulative_drugX_use_in_new_host_counts = 0

        self.cumulative_hospital_bed_use_in_days = 0
        self.cumulative_icu_bed_use_in_days = 0
        self.cumulative_ventilator_use_in_days = 0
        self.cumulative_drugX_use_in_days = 0

        self.prob_spread_virus_gamma_shape = prob_spread_virus_gamma_shape
        self.prob_spread_virus_gamma_scale = prob_spread_virus_gamma_scale
        self.prob_spread_virus_gamma_loc = prob_spread_virus_gamma_loc
        self.prob_spread_virus_gamma_magnitude_multiplier = prob_spread_virus_gamma_magnitude_multiplier
        self.prob_recover_gamma_shape = prob_recover_gamma_shape
        self.prob_recover_gamma_scale = prob_recover_gamma_scale
        self.prob_recover_gamma_loc = prob_recover_gamma_loc
        self.prob_recover_gamma_magnitude_multiplier = prob_recover_gamma_magnitude_multiplier
        self.prob_virus_kill_host_gamma_shape = prob_virus_kill_host_gamma_shape
        self.prob_virus_kill_host_gamma_scale = prob_virus_kill_host_gamma_scale
        self.prob_virus_kill_host_gamma_loc = prob_virus_kill_host_gamma_loc
        self.prob_virus_kill_host_gamma_magnitude_multiplier = prob_virus_kill_host_gamma_magnitude_multiplier
        self.prob_infectious_no_to_mild_symptom_gamma_shape = prob_infectious_no_to_mild_symptom_gamma_shape
        self.prob_infectious_no_to_mild_symptom_gamma_scale = prob_infectious_no_to_mild_symptom_gamma_scale
        self.prob_infectious_no_to_mild_symptom_gamma_loc = prob_infectious_no_to_mild_symptom_gamma_loc
        self.prob_infectious_no_to_mild_symptom_gamma_magnitude_multiplier = prob_infectious_no_to_mild_symptom_gamma_magnitude_multiplier
        self.prob_infectious_no_to_severe_symptom_gamma_shape = prob_infectious_no_to_severe_symptom_gamma_shape
        self.prob_infectious_no_to_severe_symptom_gamma_scale = prob_infectious_no_to_severe_symptom_gamma_scale
        self.prob_infectious_no_to_severe_symptom_gamma_loc = prob_infectious_no_to_severe_symptom_gamma_loc
        self.prob_infectious_no_to_severe_symptom_gamma_magnitude_multiplier = prob_infectious_no_to_severe_symptom_gamma_magnitude_multiplier
        self.prob_infectious_no_to_critical_symptom_gamma_shape = prob_infectious_no_to_critical_symptom_gamma_shape
        self.prob_infectious_no_to_critical_symptom_gamma_scale = prob_infectious_no_to_critical_symptom_gamma_scale
        self.prob_infectious_no_to_critical_symptom_gamma_loc = prob_infectious_no_to_critical_symptom_gamma_loc
        self.prob_infectious_no_to_critical_symptom_gamma_magnitude_multiplier = prob_infectious_no_to_critical_symptom_gamma_magnitude_multiplier
        self.prob_infectious_mild_to_no_symptom_gamma_shape = prob_infectious_mild_to_no_symptom_gamma_shape
        self.prob_infectious_mild_to_no_symptom_gamma_scale = prob_infectious_mild_to_no_symptom_gamma_scale
        self.prob_infectious_mild_to_no_symptom_gamma_loc = prob_infectious_mild_to_no_symptom_gamma_loc
        self.prob_infectious_mild_to_no_symptom_gamma_magnitude_multiplier = prob_infectious_mild_to_no_symptom_gamma_magnitude_multiplier
        self.prob_infectious_mild_to_severe_symptom_gamma_shape = prob_infectious_mild_to_severe_symptom_gamma_shape
        self.prob_infectious_mild_to_severe_symptom_gamma_scale = prob_infectious_mild_to_severe_symptom_gamma_scale
        self.prob_infectious_mild_to_severe_symptom_gamma_loc = prob_infectious_mild_to_severe_symptom_gamma_loc
        self.prob_infectious_mild_to_severe_symptom_gamma_magnitude_multiplier = prob_infectious_mild_to_severe_symptom_gamma_magnitude_multiplier
        self.prob_infectious_mild_to_critical_symptom_gamma_shape = prob_infectious_mild_to_critical_symptom_gamma_shape
        self.prob_infectious_mild_to_critical_symptom_gamma_scale = prob_infectious_mild_to_critical_symptom_gamma_scale
        self.prob_infectious_mild_to_critical_symptom_gamma_loc = prob_infectious_mild_to_critical_symptom_gamma_loc
        self.prob_infectious_mild_to_critical_symptom_gamma_magnitude_multiplier = prob_infectious_mild_to_critical_symptom_gamma_magnitude_multiplier
        self.prob_infectious_severe_to_no_symptom_gamma_shape = prob_infectious_severe_to_no_symptom_gamma_shape
        self.prob_infectious_severe_to_no_symptom_gamma_scale = prob_infectious_severe_to_no_symptom_gamma_scale
        self.prob_infectious_severe_to_no_symptom_gamma_loc = prob_infectious_severe_to_no_symptom_gamma_loc
        self.prob_infectious_severe_to_no_symptom_gamma_magnitude_multiplier = prob_infectious_severe_to_no_symptom_gamma_magnitude_multiplier
        self.prob_infectious_severe_to_mild_symptom_gamma_shape = prob_infectious_severe_to_mild_symptom_gamma_shape
        self.prob_infectious_severe_to_mild_symptom_gamma_scale = prob_infectious_severe_to_mild_symptom_gamma_scale
        self.prob_infectious_severe_to_mild_symptom_gamma_loc = prob_infectious_severe_to_mild_symptom_gamma_loc
        self.prob_infectious_severe_to_mild_symptom_gamma_magnitude_multiplier = prob_infectious_severe_to_mild_symptom_gamma_magnitude_multiplier
        self.prob_infectious_severe_to_critical_symptom_gamma_shape = prob_infectious_severe_to_critical_symptom_gamma_shape
        self.prob_infectious_severe_to_critical_symptom_gamma_scale = prob_infectious_severe_to_critical_symptom_gamma_scale
        self.prob_infectious_severe_to_critical_symptom_gamma_loc = prob_infectious_severe_to_critical_symptom_gamma_loc
        self.prob_infectious_severe_to_critical_symptom_gamma_magnitude_multiplier = prob_infectious_severe_to_critical_symptom_gamma_magnitude_multiplier
        self.prob_infectious_critical_to_no_symptom_gamma_shape = prob_infectious_critical_to_no_symptom_gamma_shape
        self.prob_infectious_critical_to_no_symptom_gamma_scale = prob_infectious_critical_to_no_symptom_gamma_scale
        self.prob_infectious_critical_to_no_symptom_gamma_loc = prob_infectious_critical_to_no_symptom_gamma_loc
        self.prob_infectious_critical_to_no_symptom_gamma_magnitude_multiplier = prob_infectious_critical_to_no_symptom_gamma_magnitude_multiplier
        self.prob_infectious_critical_to_mild_symptom_gamma_shape = prob_infectious_critical_to_mild_symptom_gamma_shape
        self.prob_infectious_critical_to_mild_symptom_gamma_scale = prob_infectious_critical_to_mild_symptom_gamma_scale
        self.prob_infectious_critical_to_mild_symptom_gamma_loc = prob_infectious_critical_to_mild_symptom_gamma_loc
        self.prob_infectious_critical_to_mild_symptom_gamma_magnitude_multiplier = prob_infectious_critical_to_mild_symptom_gamma_magnitude_multiplier
        self.prob_infectious_critical_to_severe_symptom_gamma_shape = prob_infectious_critical_to_severe_symptom_gamma_shape
        self.prob_infectious_critical_to_severe_symptom_gamma_scale = prob_infectious_critical_to_severe_symptom_gamma_scale
        self.prob_infectious_critical_to_severe_symptom_gamma_loc = prob_infectious_critical_to_severe_symptom_gamma_loc
        self.prob_infectious_critical_to_severe_symptom_gamma_magnitude_multiplier = prob_infectious_critical_to_severe_symptom_gamma_magnitude_multiplier

        self.prob_spread_virus_dist = GammaProbabilityGenerator(
            shape = self.prob_spread_virus_gamma_shape,
            scale = self.prob_spread_virus_gamma_scale,
            loc = self.prob_spread_virus_gamma_loc,
            magnitude_multiplier = self.prob_spread_virus_gamma_magnitude_multiplier,
        )

        self.prob_recover_dist = GammaProbabilityGenerator(
            shape = self.prob_recover_gamma_shape,
            scale = self.prob_recover_gamma_scale,
            loc = self.prob_recover_gamma_loc,
            magnitude_multiplier = self.prob_recover_gamma_magnitude_multiplier,
        )

        self.prob_virus_kill_host_dist = GammaProbabilityGenerator(
            shape = self.prob_virus_kill_host_gamma_shape,
            scale = self.prob_virus_kill_host_gamma_scale,
            loc = self.prob_virus_kill_host_gamma_loc,
            magnitude_multiplier = self.prob_virus_kill_host_gamma_magnitude_multiplier,
        )

        self.prob_infectious_no_to_mild_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_no_to_mild_symptom_gamma_shape,
            scale = self.prob_infectious_no_to_mild_symptom_gamma_scale,
            loc = self.prob_infectious_no_to_mild_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_no_to_mild_symptom_gamma_magnitude_multiplier,
        )

        self.prob_infectious_no_to_severe_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_no_to_severe_symptom_gamma_shape,
            scale = self.prob_infectious_no_to_severe_symptom_gamma_scale,
            loc = self.prob_infectious_no_to_severe_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_no_to_severe_symptom_gamma_magnitude_multiplier,
        )

        self.prob_infectious_no_to_critical_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_no_to_critical_symptom_gamma_shape,
            scale = self.prob_infectious_no_to_critical_symptom_gamma_scale,
            loc = self.prob_infectious_no_to_critical_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_no_to_critical_symptom_gamma_magnitude_multiplier,
        )

        self.prob_infectious_mild_to_no_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_mild_to_no_symptom_gamma_shape,
            scale = self.prob_infectious_mild_to_no_symptom_gamma_scale,
            loc = self.prob_infectious_mild_to_no_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_mild_to_no_symptom_gamma_magnitude_multiplier,
        )

        self.prob_infectious_mild_to_severe_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_mild_to_severe_symptom_gamma_shape,
            scale = self.prob_infectious_mild_to_severe_symptom_gamma_scale,
            loc = self.prob_infectious_mild_to_severe_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_mild_to_severe_symptom_gamma_magnitude_multiplier,
        )

        self.prob_infectious_mild_to_critical_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_mild_to_critical_symptom_gamma_shape,
            scale = self.prob_infectious_mild_to_critical_symptom_gamma_scale,
            loc = self.prob_infectious_mild_to_critical_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_mild_to_critical_symptom_gamma_magnitude_multiplier,
        )

        self.prob_infectious_severe_to_no_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_severe_to_no_symptom_gamma_shape,
            scale = self.prob_infectious_severe_to_no_symptom_gamma_scale,
            loc = self.prob_infectious_severe_to_no_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_severe_to_no_symptom_gamma_magnitude_multiplier,
        )

        self.prob_infectious_severe_to_mild_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_severe_to_mild_symptom_gamma_shape,
            scale = self.prob_infectious_severe_to_mild_symptom_gamma_scale,
            loc = self.prob_infectious_severe_to_mild_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_severe_to_mild_symptom_gamma_magnitude_multiplier,
        )

        self.prob_infectious_severe_to_critical_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_severe_to_critical_symptom_gamma_shape,
            scale = self.prob_infectious_severe_to_critical_symptom_gamma_scale,
            loc = self.prob_infectious_severe_to_critical_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_severe_to_critical_symptom_gamma_magnitude_multiplier,
        )

        self.prob_infectious_critical_to_no_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_critical_to_no_symptom_gamma_shape,
            scale = self.prob_infectious_critical_to_no_symptom_gamma_scale,
            loc = self.prob_infectious_critical_to_no_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_critical_to_no_symptom_gamma_magnitude_multiplier,
        )

        self.prob_infectious_critical_to_mild_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_critical_to_mild_symptom_gamma_shape,
            scale = self.prob_infectious_critical_to_mild_symptom_gamma_scale,
            loc = self.prob_infectious_critical_to_mild_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_critical_to_mild_symptom_gamma_magnitude_multiplier,
        )

        self.prob_infectious_critical_to_severe_symptom_dist = GammaProbabilityGenerator(
            shape = self.prob_infectious_critical_to_severe_symptom_gamma_shape,
            scale = self.prob_infectious_critical_to_severe_symptom_gamma_scale,
            loc = self.prob_infectious_critical_to_severe_symptom_gamma_loc,
            magnitude_multiplier = self.prob_infectious_critical_to_severe_symptom_gamma_magnitude_multiplier,
        )

        self.prob_recovered_no_to_mild_complication = prob_recovered_no_to_mild_complication
        self.prob_recovered_no_to_severe_complication = prob_recovered_no_to_severe_complication
        self.prob_recovered_mild_to_no_complication = prob_recovered_mild_to_no_complication
        self.prob_recovered_mild_to_severe_complication = prob_recovered_mild_to_severe_complication
        self.prob_recovered_severe_to_no_complication = prob_recovered_severe_to_no_complication
        self.prob_recovered_severe_to_mild_complication = prob_recovered_severe_to_mild_complication
        self.prob_gain_immunity = prob_gain_immunity

        self.hospital_bed_capacity_as_percent_of_population = hospital_bed_capacity_as_percent_of_population
        self.hospital_bed_cost_per_day = hospital_bed_cost_per_day
        self.hospital_bed_current_load = 0
        self.hospital_bed_use_day_tracker = 0

        self.icu_bed_capacity_as_percent_of_population = icu_bed_capacity_as_percent_of_population
        self.icu_bed_cost_per_day = icu_bed_cost_per_day
        self.icu_bed_current_load = 0
        self.icu_bed_use_day_tracker = 0

        self.ventilator_capacity_as_percent_of_population = ventilator_capacity_as_percent_of_population
        self.ventilator_cost_per_day = ventilator_cost_per_day
        self.ventilator_current_load = 0
        self.ventilator_use_day_tracker = 0

        self.drugX_capacity_as_percent_of_population = drugX_capacity_as_percent_of_population
        self.drugX_cost_per_day = drugX_cost_per_day
        self.drugX_current_load = 0
        self.drugX_use_day_tracker = 0

        self.clinical_resource = ClinicalResource(1, self,
            self.hospital_bed_capacity_as_percent_of_population, self.hospital_bed_cost_per_day,
                self.hospital_bed_current_load, self.hospital_bed_use_day_tracker,
            self.icu_bed_capacity_as_percent_of_population, self.icu_bed_cost_per_day,
                self.icu_bed_current_load, self.icu_bed_use_day_tracker,
            self.ventilator_capacity_as_percent_of_population, self.ventilator_cost_per_day,
                self.ventilator_current_load, self.ventilator_use_day_tracker,
            self.drugX_capacity_as_percent_of_population, self.drugX_cost_per_day,
                self.drugX_current_load, self.drugX_use_day_tracker,
            )

        self.testing = Testing(1, self, agent=None,
                                prob_tested_for_no_symptom=[0.005, 0.01, 0.01],
                                prob_tested_for_mild_symptom=[0.005, 0.01, 0.01],
                                prob_tested_for_severe_symptom=[0.01, 0.03, 0.05],
                                prob_tested_for_critical_symptom=[0.01, 0.03, 0.05],
                                test_sensitivity=[0.89, 0.95, 0.95], test_specificity=[0.95, 0.99, 0.99],
                                time_period=[(0, 25), (26, 60), (60, 999)], current_time=None, on_switch=True)

        self.social_distancing = SocialDistancing(1, self, edge_threshold=[0.25],
                                                  time_period=[(50, 999)], current_time=None,
                                                  on_switch=False)

        self.vaccine = Vaccine(1, self, agent=None, prob_vaccinated=[0.10],
                               vaccine_success_rate=[0.80], time_period=[(50, 999)],
                               current_time=None, on_switch=False)

        self.model_reporters_dict = {
                                'Time': return_time,
                                'Total N': return_total_n,
                                'Test done': number_test_done,
                                'Susceptible': number_susceptible,
                                'Recovered': number_recovered,
                                'Infectious': number_infectious,
                                'Dead': number_dead,
                                'Test-confirmed infectious': number_infectious_test_confirmed,
                                'Test-confirmed dead': number_dead_test_confirmed,
                                'Cumulative test done': cumulative_total_test_done,
                                'Cumulative infectious': cumulative_total_infectious,
                                'Cumulative dead': cumulative_total_dead,
                                'Cumulative test-confirmed infectious': cumulative_total_infectious_test_confirmed,
                                'Cumulative test-confirmed dead': cumulative_total_dead_test_confirmed,
                                'Rate per 1M cumulative test done': rate_cumulative_test_done,
                                'Rate per 1M cumulative infectious': rate_cumulative_infectious,
                                'Rate per 1M cumulative dead': rate_cumulative_dead,
                                'Rate per 1M cumulative test-confirmed infectious': rate_cumulative_infectious_test_confirmed,
                                'Rate per 1M cumulative test-confirmed dead': rate_cumulative_dead_test_confirmed,
                                'Infectious-no symptom': number_infectious_no_symptom,
                                'Infectious-mild symptom': number_infectious_mild_symptom,
                                'Infectious-severe symptom': number_infectious_severe_symptom,
                                'Infectious-critical symptom': number_infectious_critical_symptom,
                                'Infectious using non-ICU hospital bed': number_infectious_using_hospital_bed,
                                'Infectious using ICU hospital bed': number_infectious_using_icu_bed,
                                'Infectious using ventilator': number_infectious_using_ventilator,
                                'Recovered-no complication': number_recovered_no_complication,
                                'Recovered-mild complication': number_recovered_mild_complication,
                                'Recovered-severe complication': number_recovered_severe_complication,
                                'Recovered using DrugX': number_recovered_using_drugX,
                                'Mean R0': mean_r0,
        }
        self.datacollector = DataCollector(model_reporters=self.model_reporters_dict)

        # Create agents
        for i, node in enumerate(self.G.nodes()):
            agent = HostAgent(i, self, DiseaseHealthState.SUSCEPTIBLE, RecoveredImmunityState.TBD,
                                self.prob_recovered_no_to_mild_complication,
                                self.prob_recovered_no_to_severe_complication,
                                self.prob_recovered_mild_to_no_complication,
                                self.prob_recovered_mild_to_severe_complication,
                                self.prob_recovered_severe_to_no_complication,
                                self.prob_recovered_severe_to_mild_complication,
                                self.prob_gain_immunity,
                                self.clinical_resource, self.social_distancing,
                                self.vaccine, self.testing,
                              )
            self.schedule.add(agent)
            # Add the agent to the node
            self.grid.place_agent(agent, node)

        # Assign random weights (float: 0 to 1) to each connection
        for u, v in self.G.edges():
            self.G[u][v]['weight'] = random.random()

        # Infect some nodes
        if self.set_initial_infectious_node_seed:
            self.random.seed(self.set_initial_infectious_node_seed)
        infectious_nodes = self.random.sample(self.G.nodes(), self.initial_outbreak_size)

        for agent in self.grid.get_cell_list_contents(infectious_nodes):
            agent.disease_health_state = DiseaseHealthState.INFECTIOUS
            agent._timer_since_beginning_of_last_infection = 0

            if agent.disease_health_state is DiseaseHealthState.INFECTIOUS:
                agent.infectious_symptom_state = InfectiousSymptomState.NO_SYMPTOM

        self.running = True
        self.datacollector.collect(self)

    def ratio_infectious_susceptible(self):
        try:
            return number_disease_health_state(self, DiseaseHealthState.INFECTIOUS) / number_disease_health_state(
                self, DiseaseHealthState.SUSCEPTIBLE)
        except ZeroDivisionError:
            return math.inf

    def ratio_recovered_susceptible(self):
        try:
            return number_disease_health_state(self, DiseaseHealthState.RECOVERED) / number_disease_health_state(
                self, DiseaseHealthState.SUSCEPTIBLE)
        except ZeroDivisionError:
            return math.inf

    def ratio_dead_susceptible(self):
        try:
            return number_disease_health_state(self, DiseaseHealthState.DEAD) / number_disease_health_state(
                self, DiseaseHealthState.SUSCEPTIBLE)
        except ZeroDivisionError:
            return math.inf

    def count_total_host(self):
        return number_disease_health_state(self, DiseaseHealthState.SUSCEPTIBLE) + number_disease_health_state(
            self, DiseaseHealthState.INFECTIOUS) + \
            number_disease_health_state(self, DiseaseHealthState.DEAD) + number_disease_health_state(
            self, DiseaseHealthState.RECOVERED)

    def count_total_living_host(self):
        return number_disease_health_state(self, DiseaseHealthState.SUSCEPTIBLE) + number_disease_health_state(
            self, DiseaseHealthState.INFECTIOUS) + number_disease_health_state(self, DiseaseHealthState.RECOVERED)

    def rate_infectious(self):
        return (number_disease_health_state(self, DiseaseHealthState.INFECTIOUS) / self.count_total_host()) * self.rate_denominator

    def rate_recovered(self):
        return (number_disease_health_state(self, DiseaseHealthState.RECOVERED) / self.count_total_host()) * self.rate_denominator

    def rate_dead(self):
        return (number_disease_health_state(self, DiseaseHealthState.DEAD) / self.count_total_host()) * self.rate_denominator

    def rate_infectious_test_confirmed(self):
        return (number_disease_health_state_test_confirmed(self, DiseaseHealthState.INFECTIOUS) / self.count_total_host()) * self.rate_denominator

    def rate_dead_test_confirmed(self):
        return (number_disease_health_state_test_confirmed(self, DiseaseHealthState.DEAD) / self.count_total_host()) * self.rate_denominator

    def cumulative_total_test_done(self):
        return self.cumulative_test_done

    def cumulative_total_infectious(self):
        return self.cumulative_infectious_cases

    def cumulative_total_dead(self):
        return self.cumulative_dead_cases

    def cumulative_total_infectious_test_confirmed(self):
        return self.cumulative_infectious_test_confirmed_cases

    def cumulative_total_dead_test_confirmed(self):
        return self.cumulative_dead_test_confirmed_cases

    def cumulative_total_new_hosts_of_hospital_bed_use(self):
        return self.cumulative_hospital_bed_use_in_new_host_counts

    def cumulative_total_new_hosts_of_icu_bed_use(self):
        return self.cumulative_icu_bed_use_in_new_host_counts

    def cumulative_total_new_hosts_of_ventilator_use(self):
        return self.cumulative_ventilator_use_in_new_host_counts

    def cumulative_total_new_hosts_of_drugX_use(self):
        return self.cumulative_drugX_use_in_new_host_counts

    def cumulative_total_days_of_hospital_bed_use(self):
        return self.cumulative_hospital_bed_use_in_days

    def cumulative_total_days_of_icu_bed_use(self):
        return self.cumulative_icu_bed_use_in_days

    def cumulative_total_days_of_ventilator_use(self):
        return self.cumulative_ventilator_use_in_days

    def cumulative_total_days_of_drugX_use(self):
        return self.cumulative_drugX_use_in_days

    def cumulative_total_costs_in_hospital_bed_use(self):
        return self.cumulative_hospital_bed_use_in_days * self.hospital_bed_cost_per_day

    def cumulative_total_costs_in_icu_bed_use(self):
        return self.cumulative_icu_bed_use_in_days * self.icu_bed_cost_per_day

    def cumulative_total_costs_in_ventilator_use(self):
        return self.cumulative_ventilator_use_in_days * self.ventilator_cost_per_day

    def cumulative_total_costs_in_drugX_use(self):
        return self.cumulative_drugX_use_in_days * self.drugX_cost_per_day

    def rate_cumulative_test_done(self):
        return (self.cumulative_total_test_done() / self.count_total_host()) * self.rate_denominator

    def rate_cumulative_infectious(self):
        return (self.cumulative_total_infectious() / self.count_total_host()) * self.rate_denominator

    def rate_cumulative_dead(self):
        return (self.cumulative_total_dead() / self.count_total_host()) * self.rate_denominator

    def rate_cumulative_infectious_test_confirmed(self):
        return (self.cumulative_total_infectious_test_confirmed() / self.count_total_host()) * self.rate_denominator

    def rate_cumulative_dead_test_confirmed(self):
        return (self.cumulative_total_dead_test_confirmed() / self.count_total_host()) * self.rate_denominator

    def rate_infectious_any_symptom(self):
        return ((number_infectious_mild_symptom(self) + number_infectious_severe_symptom(self) +
                 number_infectious_critical_symptom(self)) / self.count_total_living_host()) * self.rate_denominator

    def rate_recovered_any_complication(self):
        return ((number_recovered_mild_complication(self) + number_recovered_severe_complication(self)) /
                self.count_total_living_host()) * self.rate_denominator

    def rate_infectious_using_hospital_bed(self):
        try:
            return (number_infectious_using_hospital_bed(self) / number_infectious(self) * 1000)
        except ZeroDivisionError:
            return math.inf

    def rate_infectious_using_icu_bed(self):
        try:
            return (number_infectious_using_icu_bed(self) / number_infectious(self) * 1000)
        except ZeroDivisionError:
            return math.inf

    def rate_infectious_using_ventilator(self):
        try:
            return (number_infectious_using_ventilator(self) / number_infectious(self) * 1000)
        except ZeroDivisionError:
            return math.inf

    def rate_recovered_using_drugX(self):
        try:
            return (number_recovered_using_drugX(self) / number_recovered(self) * 1000)
        except ZeroDivisionError:
            return math.inf

    def mean_age(self):
        count = 0
        total_age = 0
        for agent in self.grid.get_cell_list_contents(self.G.nodes()):
            if agent.disease_health_state is not DiseaseHealthState.DEAD:
                count += 1
                total_age += agent.age
        try:
            return total_age/count
        except ZeroDivisionError:
            return math.inf

    def proportion_sex(self):
        count = 0
        total_male = 0
        total_female = 0
        for agent in self.grid.get_cell_list_contents(self.G.nodes()):
            if agent.disease_health_state is not DiseaseHealthState.DEAD:
                count += 1
                if agent.sex is 'M':
                    total_male += 1
                elif agent.sex is 'F':
                    total_female += 1
        try:
            return {'M': total_male/count, 'F': total_female/count}
        except ZeroDivisionError:
            return {'M': math.inf, 'F': math.inf}

    def mean_r0(self):
        '''WARNING: Need triple check, does not look correct right now'''
        number_infectious_active_in_last_n_time_units = 0
        number_new_infection_in_last_n_time_units = 0
        last_n_time_unit = self._last_n_time_unit_for_mean_r0

        for agent, content in self.all_agents_new_infection_tracker.items():
            for time_of_new_infection, number_of_new_infection in content.items():
                initial_time = self._current_timer - last_n_time_unit
                if initial_time >= 0:
                    if int(time_of_new_infection) in range(initial_time, self._current_timer):
                        number_infectious_active_in_last_n_time_units += 1
                        number_new_infection_in_last_n_time_units += number_of_new_infection
                else:
                    number_infectious_active_in_last_n_time_units += 1
                    number_new_infection_in_last_n_time_units += number_of_new_infection

        try:
            return number_new_infection_in_last_n_time_units / number_infectious_active_in_last_n_time_units
        except ZeroDivisionError:
            return 0

    def step(self):
        self._current_timer += 1
        self.schedule.step()
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in range(n):
            self.step()