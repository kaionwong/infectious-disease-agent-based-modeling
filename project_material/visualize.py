import math
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import TextElement

from .model.network import HostNetwork
from .model.state import DiseaseHealthState, RecoveredImmunityState, number_susceptible, \
        number_infectious, number_recovered, number_dead, number_infectious_no_symptom, \
        number_infectious_mild_symptom, number_infectious_severe_symptom, number_infectious_critical_symptom, \
        number_recovered_no_complication, number_recovered_mild_complication, number_recovered_severe_complication, \
        number_infectious_test_confirmed, number_dead_test_confirmed

def network_portrayal(G):
    # The model ensures there is always 1 agent per node
    def node_color(agent):
        return {
            DiseaseHealthState.SUSCEPTIBLE: '#008000',
            DiseaseHealthState.INFECTIOUS: '#FF0000',
            DiseaseHealthState.DEAD: '#000000',
        }.get(agent.disease_health_state, '#00c5cf')

    def edge_color(agent1, agent2):
        if DiseaseHealthState.DEAD in (agent1.disease_health_state, agent2.disease_health_state):
            return '#FFFFFF'
        if DiseaseHealthState.RECOVERED in (agent1.disease_health_state, agent2.disease_health_state):
            if RecoveredImmunityState.WITH_IMMUNITY in (agent1.recovered_immunity_state,
                                                        agent2.recovered_immunity_state):
                return '#bbe7f2'
        else:
            return '#757575'

    def edge_width(agent1, agent2):
        if DiseaseHealthState.DEAD in (agent1.disease_health_state, agent2.disease_health_state):
            return 1
        else:
            return 2

    def get_agents(source, target):
        return G.nodes[source]['agent'][0], G.nodes[target]['agent'][0]

    portrayal = dict()
    portrayal['nodes'] = [{'size': 6,
                           'color': node_color(agents[0]),
                           'tooltip': "id: {}<br>state: {}".format(agents[0].unique_id, agents[0].disease_health_state.name),
                           }
                          for (_, agents) in G.nodes.data('agent')]

    portrayal['edges'] = [{'source': source,
                           'target': target,
                           'color': edge_color(*get_agents(source, target)),
                           'width': edge_width(*get_agents(source, target)),
                           }
                          for (source, target) in G.edges]

    return portrayal

network = NetworkModule(network_portrayal, 700, 730, library='d3')

chart_all_counts = ChartModule([
                        {'Label': 'Susceptible', 'Color': '#008000'},
                        {'Label': 'Infectious', 'Color': '#FF0000'},
                        {'Label': 'Recovered', 'Color': '#00C5CD'},
                        {'Label': 'Dead', 'Color': '#000000'},
                     ])

chart_infectious_and_dead = ChartModule([
                        {'Label': 'Infectious', 'Color': '#FF0000'},
                        {'Label': 'Dead', 'Color': '#000000'},
                     ])

chart_infectious_and_dead_test_confirmed = ChartModule([
                        {'Label': 'Test-confirmed infectious', 'Color': '#FF0000'},
                        {'Label': 'Test-confirmed dead', 'Color': '#000000'},
                     ])

chart_rate_cumulative_infectious = ChartModule([
                        {'Label': 'Rate per 1M cumulative infectious', 'Color': '#cf0e52'},
                        {'Label': 'Rate per 1M cumulative test-confirmed infectious', 'Color': '#432c4a'},
                     ])

chart_rate_cumulative_dead = ChartModule([
                        {'Label': 'Rate per 1M cumulative dead', 'Color': '#b5a7d1'},
                        {'Label': 'Rate per 1M cumulative test-confirmed dead', 'Color': '#221240'},
                     ])

chart_infectious_symptom_state = ChartModule([
                        {'Label': 'Infectious-no symptom', 'Color': '#ff8f8f'},
                        {'Label': 'Infectious-mild symptom', 'Color': '#ff0000'},
                        {'Label': 'Infectious-severe symptom', 'Color': '#6e0000'},
                        {'Label': 'Infectious-critical symptom', 'Color': '#2b0000'},
                     ])

chart_recovered_complication_state = ChartModule([
                        {'Label': 'Recovered-no complication', 'Color': '#74afc4'},
                        {'Label': 'Recovered-mild complication', 'Color': '#00c5cf'},
                        {'Label': 'Recovered-severe complication', 'Color': '#006469'},
                     ])

chart_clinical_service_use_for_infectious = ChartModule([
                        {'Label': 'Infectious using ventilator', 'Color': '#102ec7'},
                        {'Label': 'Infectious using non-ICU hospital bed', 'Color': '#ba09b7'},
                        {'Label': 'Infectious using ICU hospital bed', 'Color': '#b88009'},
                     ])

chart_clinical_service_use_for_recovered = ChartModule([
                        {'Label': 'Recovered using DrugX', 'Color': '#0e998b'},
                     ])

chart_r0 = ChartModule([
                        {'Label': 'Mean R0', 'Color': '#000000'},
                     ])

class MainTextElement(TextElement):
    def render(self, model):
        num_susceptible_text = str(number_susceptible(model))
        num_infectious_text = str(number_infectious(model))
        num_recovered_text = str(number_recovered(model))
        num_dead_text = str(number_dead(model))

        num_infectious_test_confirmed_text = str(number_infectious_test_confirmed(model))
        num_dead_test_confirmed_text = str(number_dead_test_confirmed(model))

        cumulative_total_infectious_text =  '{0:.0f}'.format(model.cumulative_total_infectious())
        cumulative_total_dead_text =  '{0:.0f}'.format(model.cumulative_total_dead())

        cumulative_total_new_hosts_of_hospital_bed_use_text =  '{0:.0f}'.format(model.cumulative_total_new_hosts_of_hospital_bed_use())
        cumulative_total_new_hosts_of_icu_bed_use_text =  '{0:.0f}'.format(model.cumulative_total_new_hosts_of_icu_bed_use())
        cumulative_total_new_hosts_of_ventilator_use_text =  '{0:.0f}'.format(model.cumulative_total_new_hosts_of_ventilator_use())
        cumulative_total_new_hosts_of_drugX_use_text = '{0:.0f}'.format(model.cumulative_total_new_hosts_of_drugX_use())

        cumulative_total_days_of_hospital_bed_use_text =  '{0:.0f}'.format(model.cumulative_total_days_of_hospital_bed_use())
        cumulative_total_days_of_icu_bed_use_text = '{0:.0f}'.format(model.cumulative_total_days_of_icu_bed_use())
        cumulative_total_days_of_ventilator_use_text = '{0:.0f}'.format(model.cumulative_total_days_of_ventilator_use())
        cumulative_total_days_of_drugX_use_text = '{0:.0f}'.format(model.cumulative_total_days_of_drugX_use())

        cumulative_total_costs_in_hospital_bed_use_text =  '{0:.0f}'.format(model.cumulative_total_costs_in_hospital_bed_use())
        cumulative_total_costs_in_icu_bed_use_text = '{0:.0f}'.format(model.cumulative_total_costs_in_icu_bed_use())
        cumulative_total_costs_in_ventilator_use_text = '{0:.0f}'.format(model.cumulative_total_costs_in_ventilator_use())
        cumulative_total_costs_in_drugX_use_text = '{0:.0f}'.format(model.cumulative_total_costs_in_drugX_use())

        num_infectious_no_symptom_text = str(number_infectious_no_symptom(model))
        num_infectious_mild_symptom_text = str(number_infectious_mild_symptom(model))
        num_infectious_severe_symptom_text = str(number_infectious_severe_symptom(model))
        num_infectious_critical_symptom_text = str(number_infectious_critical_symptom(model))

        num_recovered_no_complication_text = str(number_recovered_no_complication(model))
        num_recovered_mild_complication_text = str(number_recovered_mild_complication(model))
        num_recovered_severe_complication_text = str(number_recovered_severe_complication(model))

        ratio_infectious_susceptible = model.ratio_infectious_susceptible()
        ratio_recovered_susceptible = model.ratio_recovered_susceptible()
        ratio_dead_susceptible = model.ratio_dead_susceptible()

        ratio_infectious_susceptible_text = '&infin;' if ratio_infectious_susceptible is math.inf else '{0:.2f}'.format(
            ratio_infectious_susceptible)
        ratio_recovered_susceptible_text = '&infin;' if ratio_recovered_susceptible is math.inf else '{0:.2f}'.format(
            ratio_recovered_susceptible)
        ratio_dead_susceptible_text = '&infin;' if ratio_dead_susceptible is math.inf else '{0:.2f}'.format(
            ratio_dead_susceptible)

        rate_infectious_text = '{0:.2f}'.format(model.rate_infectious())
        rate_recovered_text = '{0:.2f}'.format(model.rate_recovered())
        rate_dead_text = '{0:.2f}'.format(model.rate_dead())
        rate_infectious_test_confirmed_text = '{0:.2f}'.format(model.rate_infectious_test_confirmed())
        rate_dead_test_confirmed_text = '{0:.2f}'.format(model.rate_dead_test_confirmed())
        rate_infectious_any_symptom_text = '{0:.2f}'.format(model.rate_infectious_any_symptom())
        rate_recovered_any_complication_text = '{0:.2f}'.format(model.rate_recovered_any_complication())

        rate_infectious_using_hospital_bed_text = '{0:.2f}'.format(model.rate_infectious_using_hospital_bed())
        rate_infectious_using_icu_bed_text = '{0:.2f}'.format(model.rate_infectious_using_icu_bed())
        rate_infectious_using_ventilator_text = '{0:.2f}'.format(model.rate_infectious_using_ventilator())
        rate_recovered_using_drugX_text = '{0:.2f}'.format(model.rate_recovered_using_drugX())

        mean_age_text = '{0:.2f}'.format(model.mean_age())
        mean_r0_text = '{0:.2f}'.format(model.mean_r0())

        proportion_male_text = '{0:.2f}'.format(model.proportion_sex()['M'] * 100)
        proportion_female_text = '{0:.2f}'.format(model.proportion_sex()['F'] * 100)

        return '<p>&nbsp;</p> \
                <br><div align="center"><b>===== Current summary =====</b></div> \
                <br><u>Demographics</u>\
                <br>Mean age: {}\
                <br>Sex: M({}%), F({}%)\
                <br> \
                <br><u>Current count</u> \
                <br>Susceptible cases: {}\
        		<br>Infectious cases: {}\
        		<br>Recovered cases: {}\
                <br>Dead cases: {}\
                <br> \
                <br>Test-confirmed infectious cases: {}\
                <br>Test-confirmed dead cases: {}\
                <br> \
                <br>Infectious cases with no symptom: {}\
                <br>Infectious cases with mild symptom: {}\
                <br>Infectious cases with severe symptom: {}\
                <br>Infectious cases with critical symptom: {}\
                <br> \
                <br>Recovered cases with no complication: {}\
                <br>Recovered cases with mild complication: {}\
                <br>Recovered cases with severe complication: {}\
                <br> \
                <br><u>Ratio</u>\
                <br>Infectious/susceptible ratio: {}\
                <br>Recovered/susceptible ratio: {}\
                <br>Dead/susceptible ratio: {}\
                <br> \
                <br><u>Rate</u>\
                <br>R0: {}\
                <br> \
                <br>Rate of infectious cases per 1M initial population: {}\
                <br>Rate of recovered cases per 1M initial population: {}\
                <br>Rate of dead cases per 1M initial population: {}\
                <br> \
                <br>Rate of test-confirmed infectious cases per 1M initial population: {}\
                <br>Rate of test-confirmed dead cases per 1M initial population: {}\
                <br> \
                <br>Rate of infectious cases with any symptom per 1M living population: {}\
                <br>Rate of recovered cases with any complication per 1M living population: {}\
                <br> \
                <br>Rate of hospital bed uses per 1K infectious hosts: {} \
                <br>Rate of ICU bed uses per 1K infectious hosts {} \
                <br>Rate of ventilator uses per 1K infectious hosts: {} \
                <br>Rate of DrugX uses per 1K recovered hosts: {} \
                <p>&nbsp;</p> \
                <br><div align="center"><b>===== Cumulative summary =====</b></div>\
                <br><u>Cumulative count</u> \
                <br>Infections: {} \
                <br>Deaths: {} \
                <br> \
                <br><u>Cumulative total number of incident hosts for each clinical service</u> \
                <br>Hospital bed uses: {} \
                <br>ICU bed uses: {} \
                <br>Ventilator uses: {} \
                <br>DrugX uses: {} \
                <br> \
                <br><u>Cumulative total number of days used for each clinical service</u> \
                <br>Hospital bed uses: {} \
                <br>ICU bed uses: {} \
                <br>Ventilator uses: {} \
                <br>DrugX uses: {} \
                <br> \
                <br><u>Cumulative total costs for each clinical service</u> \
                <br>Hospital bed uses: ${} \
                <br>ICU bed uses: ${} \
                <br>Ventilator uses: ${} \
                <br>DrugX uses: ${} \
                <p>&nbsp;</p> \
               '.format(mean_age_text, proportion_male_text, proportion_female_text,
                        num_susceptible_text, num_infectious_text, num_recovered_text, num_dead_text,
                        num_infectious_test_confirmed_text,num_dead_test_confirmed_text,
                        num_infectious_no_symptom_text, num_infectious_mild_symptom_text, num_infectious_severe_symptom_text,
                        num_infectious_critical_symptom_text,
                        num_recovered_no_complication_text, num_recovered_mild_complication_text,
                        num_recovered_severe_complication_text,
                        ratio_infectious_susceptible_text, ratio_recovered_susceptible_text, ratio_dead_susceptible_text,
                        mean_r0_text,
                        rate_infectious_text, rate_recovered_text, rate_dead_text, rate_infectious_test_confirmed_text,
                        rate_dead_test_confirmed_text, rate_infectious_any_symptom_text,
                        rate_recovered_any_complication_text,
                        rate_infectious_using_hospital_bed_text, rate_infectious_using_icu_bed_text,
                        rate_infectious_using_ventilator_text, rate_recovered_using_drugX_text,
                        cumulative_total_infectious_text, cumulative_total_dead_text,
                        cumulative_total_new_hosts_of_hospital_bed_use_text, cumulative_total_new_hosts_of_icu_bed_use_text,
                        cumulative_total_new_hosts_of_ventilator_use_text, cumulative_total_new_hosts_of_drugX_use_text,
                        cumulative_total_days_of_hospital_bed_use_text, cumulative_total_days_of_icu_bed_use_text,
                        cumulative_total_days_of_ventilator_use_text,
                        cumulative_total_days_of_drugX_use_text,
                        cumulative_total_costs_in_hospital_bed_use_text, cumulative_total_costs_in_icu_bed_use_text,
                        cumulative_total_costs_in_ventilator_use_text, cumulative_total_costs_in_drugX_use_text
                        )

class SupportTextElement(TextElement):
    def __init__(self, my_text='', first_line_break=None):
        self.my_text = my_text
        self.first_line_break = first_line_break

    def render(self, model):
        if self.first_line_break:
            return '<br>'+self.my_text
        else:
            return self.my_text

model_params = {
    'num_nodes': UserSettableParameter(
                'slider', 'Number of hosts', 2000, 5, 5000, 10, # 10000 is about max for computing power
                description='Number of hosts at the beginning of the model'),
    'avg_node_degree': UserSettableParameter(
                'slider', 'Average connections per host', 8, 1, 20, 0.5,
                description='Average number of node per host'),
    'initial_outbreak_size': UserSettableParameter(
                'slider', 'Initial number of infectious hosts', 2, 1, 25, 1,
                description='Initial number of infectious hosts'),

    'prob_spread_virus_gamma_shape': UserSettableParameter(
                'slider', 'Shape param for virus spread prob', 2, 1.0, 30, 0.005,
                description='Set shape param in gamma probability for an infectious host to infect a neighboring host'),
    'prob_spread_virus_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for virus spread prob', 3, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for an infectious host to infect a neighboring host'),
    'prob_spread_virus_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for virus spread prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for an infectious host to infect a neighboring host'),
    'prob_spread_virus_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for virus spread prob', 0.35, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for an infectious host to infect a neighboring host'),

    'prob_recover_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for recovery prob', 3, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for an infectious host to recovered'),
    'prob_recover_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for recovery prob', 5, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for an infectious host to recovered'),
    'prob_recover_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for recovery prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for an infectious host to recovered'),
    'prob_recover_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for recovery prob', 0.45, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for an infectious host to recovered'),

    'prob_virus_kill_host_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for host to die from virus prob', 5.2, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for an infectious host to die from virus'),
    'prob_virus_kill_host_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for host to die from virus prob', 3.2, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for an infectious host to die from virus'),
    'prob_virus_kill_host_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for host to die from virus prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for an infectious host to die from virus'),
    'prob_virus_kill_host_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for host to die from virus prob', 0.05, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for an infectious host to die from virus'),

    'prob_infectious_no_to_mild_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from no to mild symptom prob', 4.1, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for an asymptomatic infectious host to develop mild symptom'),
    'prob_infectious_no_to_mild_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from no to mild symptom prob', 1, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for an asymptomatic infectious host to develop mild symptom'),
    'prob_infectious_no_to_mild_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from no to mild symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for an asymptomatic infectious host to develop mild symptom'),
    'prob_infectious_no_to_mild_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from no to mild symptom prob', 0.75, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for an asymptomatic infectious host to develop mild symptom'),

    'prob_infectious_no_to_severe_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from no to severe symptom prob', 1, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for an asymptomatic infectious host to develop severe symptom'),
    'prob_infectious_no_to_severe_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from no to severe symptom prob', 2, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for an asymptomatic infectious host to develop severe symptom'),
    'prob_infectious_no_to_severe_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from no to severe symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for an asymptomatic infectious host to develop severe symptom'),
    'prob_infectious_no_to_severe_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from no to severe symptom prob', 0.1, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for an asymptomatic infectious host to develop severe symptom'),

    'prob_infectious_no_to_critical_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from no to critical symptom prob', 1, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for an asymptomatic infectious host to develop critical symptom'),
    'prob_infectious_no_to_critical_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from no to critical symptom prob', 2.8, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for an asymptomatic infectious host to develop critical symptom'),
    'prob_infectious_no_to_critical_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from no to critical symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for an asymptomatic infectious host to develop critical symptom'),
    'prob_infectious_no_to_critical_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from no to critical symptom prob', 0.15, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for an asymptomatic infectious host to develop critical symptom'),

    'prob_infectious_mild_to_no_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from mild to no symptom prob', 3, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for a mild symptom infectious host to be asymptomatic'),
    'prob_infectious_mild_to_no_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from mild to no symptom prob', 3, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for a mild symptom infectious host to be asymptomatic'),
    'prob_infectious_mild_to_no_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from mild to no symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for a mild symptom infectious host to be asymptomatic'),
    'prob_infectious_mild_to_no_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from mild to no symptom prob', 0.25, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for a mild symptom infectious host to be asymptomatic'),

    'prob_infectious_mild_to_severe_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from mild to severe symptom prob', 4.9, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for a mild symptom infectious host to have severe symptoms'),
    'prob_infectious_mild_to_severe_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from mild to severe symptom prob', 2.2, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for a mild symptom infectious host to have severe symptoms'),
    'prob_infectious_mild_to_severe_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from mild to severe symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for a mild symptom infectious host to have severe symptoms'),
    'prob_infectious_mild_to_severe_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from mild to severe symptom prob', 0.11, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for a mild symptom infectious host to have severe symptoms'),

    'prob_infectious_mild_to_critical_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from mild to critical symptom prob', 3.3, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for a mild symptom infectious host to have critical symptoms'),
    'prob_infectious_mild_to_critical_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from mild to critical symptom prob', 3.1, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for a mild symptom infectious host to have critical symptoms'),
    'prob_infectious_mild_to_critical_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from mild to critical symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for a mild symptom infectious host to have critical symptoms'),
    'prob_infectious_mild_to_critical_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from mild to critical symptom prob', 0.11, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for a mild symptom infectious host to have critical symptoms'),

    'prob_infectious_severe_to_no_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from severe to no symptom prob', 3, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for a severe symptom infectious host to be asymptomatic'),
    'prob_infectious_severe_to_no_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from severe to no symptom prob', 2, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for a severe symptom infectious host to be asymptomatic'),
    'prob_infectious_severe_to_no_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from severe to no symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for a severe symptom infectious host to be asymptomatic'),
    'prob_infectious_severe_to_no_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from severe to no symptom prob', 0.001, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for a severe symptom infectious host to be asymptomatic'),

    'prob_infectious_severe_to_mild_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from severe to mild symptom prob', 5, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for a severe symptom infectious host to have mild symptoms'),
    'prob_infectious_severe_to_mild_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from severe to mild symptom prob', 3, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for a severe symptom infectious host to have mild symptoms'),
    'prob_infectious_severe_to_mild_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from severe to mild symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for a severe symptom infectious host to have mild symptoms'),
    'prob_infectious_severe_to_mild_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from severe to mild symptom prob', 0.001, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for a severe symptom infectious host to have mild symptoms'),

    'prob_infectious_severe_to_critical_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from severe to critical symptom prob', 7, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for a severe symptom infectious host to have critical symptoms'),
    'prob_infectious_severe_to_critical_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from severe to critical symptom prob', 3, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for a severe symptom infectious host to have critical symptoms'),
    'prob_infectious_severe_to_critical_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from severe to critical symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for a severe symptom infectious host to have critical symptoms'),
    'prob_infectious_severe_to_critical_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from severe to critical symptom prob', 0.01, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for a severe symptom infectious host to have critical symptoms'),

    'prob_infectious_critical_to_no_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from critical to no symptom prob', 7, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for a critical symptom infectious host to be asymptomatic'),
    'prob_infectious_critical_to_no_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from critical to no symptom prob', 1, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for a critical symptom infectious host to be asymptomatic'),
    'prob_infectious_critical_to_no_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from critical to no symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for a critical symptom infectious host to be asymptomatic'),
    'prob_infectious_critical_to_no_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from critical to no symptom prob', 0.001, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for a critical symptom infectious host to be asymptomatic'),

    'prob_infectious_critical_to_mild_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from critical to mild symptom prob', 4, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for a critical symptom infectious host to have mild symptoms'),
    'prob_infectious_critical_to_mild_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from critical to mild symptom prob', 2, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for a critical symptom infectious host to have mild symptoms'),
    'prob_infectious_critical_to_mild_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from critical to mild symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for a critical symptom infectious host to have mild symptoms'),
    'prob_infectious_critical_to_mild_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from critical to mild symptom prob', 0.001, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for a critical symptom infectious host to have mild symptoms'),

    'prob_infectious_critical_to_severe_symptom_gamma_shape': UserSettableParameter(
        'slider', 'Shape param for infectious from critical to severe symptom prob', 5, 1.0, 30, 0.005,
        description='Set shape param in gamma probability for a critical symptom infectious host to have severe symptoms'),
    'prob_infectious_critical_to_severe_symptom_gamma_scale': UserSettableParameter(
        'slider', 'Scale param for infectious from critical to severe symptom prob', 2, 0.0, 30, 0.005,
        description='Set scale param in gamma probability for a critical symptom infectious host to have severe symptoms'),
    'prob_infectious_critical_to_severe_symptom_gamma_loc': UserSettableParameter(
        'slider', 'Loc param for infectious from critical to severe symptom prob', 0, 0.0, 30, 0.5,
        description='Set loc param in gamma probability for a critical symptom infectious host to have severe symptoms'),
    'prob_infectious_critical_to_severe_symptom_gamma_magnitude_multiplier': UserSettableParameter(
        'slider', 'MM param for infectious from critical to severe symptom prob', 0.25, 0.0, 1, 0.001,
        description='Set magnitude multiplier param in gamma probability for a critical symptom infectious host to have severe symptoms'),

    'prob_recovered_no_to_mild_complication': UserSettableParameter(
                'slider', 'Prob from no to mild complications', 0.016, 0.0, 1.0, 0.001,
                description='Probability from no to mild complications in recovered hosts'),
    'prob_recovered_no_to_severe_complication': UserSettableParameter(
                'slider', 'Prob from no to severe complications', 0.00, 0.0, 1.0, 0.001,
                description='Probability from no to severe complications in recovered hosts'),
    'prob_recovered_mild_to_no_complication': UserSettableParameter(
                'slider', 'Prob from mild to no complications', 0.02, 0.0, 1.0, 0.001,
                description='Probability from mild to no complications in recovered hosts'),
    'prob_recovered_mild_to_severe_complication': UserSettableParameter(
                'slider', 'Prob from mild to severe complications', 0.02, 0.0, 1.0, 0.001,
                description='Probability from mild to severe complications in recovered hosts'),
    'prob_recovered_severe_to_no_complication': UserSettableParameter(
                'slider', 'Prob from severe to no complications', 0.001, 0.0, 1.0, 0.001,
                description='Probability from severe to no complications in infectious hosts'),
    'prob_recovered_severe_to_mild_complication': UserSettableParameter(
                'slider', 'Prob from severe to mild complications', 0.001, 0.0, 1.0, 0.001,
                description='Probability from severe to mild complications in recovered hosts'),

    'prob_gain_immunity': UserSettableParameter(
        'slider', 'Prob to gain immunity', 0.005, 0.0, 1, 0.001,
        description='Probability a recovered host gains immunity against the virus'),

    'hospital_bed_capacity_as_percent_of_population': UserSettableParameter(
                'slider', 'Hospital bed capacity', 0.10, 0.0, 1.0, 0.05,
                description='Hospital bed capacity as percent of population'),
    'hospital_bed_cost_per_day': UserSettableParameter(
                'slider', 'Hospitalization-related cost', 1000, 0, 100000, 500,
                description='Hospitalization-related cost per day'),

    'icu_bed_capacity_as_percent_of_population': UserSettableParameter(
                'slider', 'ICU bed capacity', 0.10, 0.0, 1.0, 0.05,
                description='ICU bed capacity as percent of population'),
    'icu_bed_cost_per_day': UserSettableParameter(
                'slider', 'ICU-related cost', 2000, 0, 100000, 500,
                description='ICU-related cost per day'),

    'ventilator_capacity_as_percent_of_population': UserSettableParameter(
                'slider', 'Ventilator capacity', 0.10, 0.0, 1.0, 0.05,
                description='Ventilator capacity as percent of population'),
    'ventilator_cost_per_day': UserSettableParameter(
                'slider', 'Ventilator-related cost', 100, 0, 5000, 100,
                description='Ventilator-related cost per day'),

    'drugX_capacity_as_percent_of_population': UserSettableParameter(
        'slider', 'DrugX capacity', 0.3, 0.0, 200, 0.05,
        description='DrugX capacity as percent of population'),
    'drugX_cost_per_day': UserSettableParameter(
        'slider', 'DrugX-related cost', 20, 0, 5000, 100,
        description='DrugX-related cost per day'),
    }

def make_server(graphics_option, server_port=8521):
    model_label = 'Infectious Disease Simulator'
    full_display = [
        network,
        SupportTextElement('Title: Mean R0', True),
                            chart_r0,
        SupportTextElement('Title: Daily count of hosts with different disease states', True),
                            chart_all_counts,
        SupportTextElement('Title: Daily count of hosts with different disease states', True),
                            chart_infectious_and_dead,
        SupportTextElement('Title: Daily count of hosts with different test-confirmed disease states', True),
                            chart_infectious_and_dead_test_confirmed,
        SupportTextElement('Title: Rate per 1M for cumulative infectious cases', True),
                            chart_rate_cumulative_infectious,
        SupportTextElement('Title: Rate per 1M for cumulative dead cases', True),
                            chart_rate_cumulative_dead,
        SupportTextElement('Title: Daily count of infectious hosts with different symptom states', True),
                            chart_infectious_symptom_state,
        SupportTextElement('Title: Daily count of infectious hosts with different clinical service usage', True),
                            chart_clinical_service_use_for_infectious,
        SupportTextElement('Title: Daily count of recovered hosts with different complication states', True),
                            chart_recovered_complication_state,
        SupportTextElement('Title: Daily count of recovered hosts with different clinical service usage', True),
                            chart_clinical_service_use_for_recovered,
        MainTextElement(),
    ]

    if graphics_option is 'default':
        current_server = ModularServer(HostNetwork, [network, SupportTextElement(), chart_all_counts,
                                                     MainTextElement()], model_label, model_params)
    elif graphics_option is 'full':
        current_server = ModularServer(HostNetwork, full_display, model_label, model_params)
    elif graphics_option is 'full_without_network_graph':
        current_server = ModularServer(HostNetwork, full_display[1:], model_label, model_params)
    elif graphics_option is 'text_only':
        current_server = ModularServer(HostNetwork, [MainTextElement()], model_label, model_params)

    current_server.port = server_port
    return current_server

server = make_server('full_without_network_graph')