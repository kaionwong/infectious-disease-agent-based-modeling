import sys
import os
import math
import datetime
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import scipy.stats as stats
from mesa.batchrunner import BatchRunner, BatchRunnerMP
from mesa.datacollection import DataCollector
from project_material.model.network import HostNetwork

class CustomBatchRunner(BatchRunner):
    def run_model(self, model):
        while model.schedule.steps < self.max_steps:
            model.step()

def track_params(model):
    return (
        model.num_nodes,
        model.avg_node_degree,
        model.initial_outbreak_size,
        model.prob_spread_virus_gamma_shape,
        model.prob_spread_virus_gamma_scale,
        model.prob_spread_virus_gamma_loc,
        model.prob_spread_virus_gamma_magnitude_multiplier,
        model.prob_recover_gamma_shape,
        model.prob_recover_gamma_scale,
        model.prob_recover_gamma_loc,
        model.prob_recover_gamma_magnitude_multiplier,
        model.prob_virus_kill_host_gamma_shape,
        model.prob_virus_kill_host_gamma_scale,
        model.prob_virus_kill_host_gamma_loc,
        model.prob_virus_kill_host_gamma_magnitude_multiplier,
        model.prob_infectious_no_to_mild_symptom_gamma_shape,
        model.prob_infectious_no_to_mild_symptom_gamma_scale,
        model.prob_infectious_no_to_mild_symptom_gamma_loc,
        model.prob_infectious_no_to_mild_symptom_gamma_magnitude_multiplier,
        model.prob_infectious_no_to_severe_symptom_gamma_shape,
        model.prob_infectious_no_to_severe_symptom_gamma_scale,
        model.prob_infectious_no_to_severe_symptom_gamma_loc,
        model.prob_infectious_no_to_severe_symptom_gamma_magnitude_multiplier,
        model.prob_infectious_no_to_critical_symptom_gamma_shape,
        model.prob_infectious_no_to_critical_symptom_gamma_scale,
        model.prob_infectious_no_to_critical_symptom_gamma_loc,
        model.prob_infectious_no_to_critical_symptom_gamma_magnitude_multiplier,
        model.prob_infectious_mild_to_no_symptom_gamma_shape,
        model.prob_infectious_mild_to_no_symptom_gamma_scale,
        model.prob_infectious_mild_to_no_symptom_gamma_loc,
        model.prob_infectious_mild_to_no_symptom_gamma_magnitude_multiplier,
        model.prob_infectious_mild_to_severe_symptom_gamma_shape,
        model.prob_infectious_mild_to_severe_symptom_gamma_scale,
        model.prob_infectious_mild_to_severe_symptom_gamma_loc,
        model.prob_infectious_mild_to_severe_symptom_gamma_magnitude_multiplier,
        model.prob_infectious_mild_to_critical_symptom_gamma_shape,
        model.prob_infectious_mild_to_critical_symptom_gamma_scale,
        model.prob_infectious_mild_to_critical_symptom_gamma_loc,
        model.prob_infectious_mild_to_critical_symptom_gamma_magnitude_multiplier,
        model.prob_infectious_severe_to_no_symptom_gamma_shape,
        model.prob_infectious_severe_to_no_symptom_gamma_scale,
        model.prob_infectious_severe_to_no_symptom_gamma_loc,
        model.prob_infectious_severe_to_no_symptom_gamma_magnitude_multiplier,
        model.prob_infectious_severe_to_mild_symptom_gamma_shape,
        model.prob_infectious_severe_to_mild_symptom_gamma_scale,
        model.prob_infectious_severe_to_mild_symptom_gamma_loc,
        model.prob_infectious_severe_to_mild_symptom_gamma_magnitude_multiplier,
        model.prob_infectious_severe_to_critical_symptom_gamma_shape,
        model.prob_infectious_severe_to_critical_symptom_gamma_scale,
        model.prob_infectious_severe_to_critical_symptom_gamma_loc,
        model.prob_infectious_severe_to_critical_symptom_gamma_magnitude_multiplier,
        model.prob_infectious_critical_to_no_symptom_gamma_shape,
        model.prob_infectious_critical_to_no_symptom_gamma_scale,
        model.prob_infectious_critical_to_no_symptom_gamma_loc,
        model.prob_infectious_critical_to_no_symptom_gamma_magnitude_multiplier,
        model.prob_infectious_critical_to_mild_symptom_gamma_shape,
        model.prob_infectious_critical_to_mild_symptom_gamma_scale,
        model.prob_infectious_critical_to_mild_symptom_gamma_loc,
        model.prob_infectious_critical_to_mild_symptom_gamma_magnitude_multiplier,
        model.prob_infectious_critical_to_severe_symptom_gamma_shape,
        model.prob_infectious_critical_to_severe_symptom_gamma_scale,
        model.prob_infectious_critical_to_severe_symptom_gamma_loc,
        model.prob_infectious_critical_to_severe_symptom_gamma_magnitude_multiplier,
        model.prob_recovered_no_to_mild_complication,
        model.prob_recovered_no_to_severe_complication,
        model.prob_recovered_mild_to_no_complication,
        model.prob_recovered_mild_to_severe_complication,
        model.prob_recovered_severe_to_no_complication,
        model.prob_recovered_severe_to_mild_complication,
        model.prob_gain_immunity,
        model.hospital_bed_capacity_as_percent_of_population,
        model.hospital_bed_cost_per_day,
        model.icu_bed_capacity_as_percent_of_population,
        model.icu_bed_cost_per_day,
        model.ventilator_capacity_as_percent_of_population,
        model.ventilator_cost_per_day,
        model.drugX_capacity_as_percent_of_population,
        model.drugX_cost_per_day,
    )

def track_run(model):
    return model.uid

class BatchHostNetwork(HostNetwork):
    # id generator to track run number in batch run data
    id_gen = itertools.count(1)

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

        super().__init__(
            num_nodes, avg_node_degree, initial_outbreak_size,

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
        )

        self.model_reporters_dict.update({'Model params': track_params, 'Run': track_run})
        self.datacollector = DataCollector(model_reporters=self.model_reporters_dict)

# parameter lists for each parameter to be tested in batch run
br_params = {
    'num_nodes': [500],
    'avg_node_degree': [10],
    'initial_outbreak_size': [2],

    'prob_spread_virus_gamma_shape': [1],
    'prob_spread_virus_gamma_scale': [3],
    'prob_spread_virus_gamma_loc': [0],
    'prob_spread_virus_gamma_magnitude_multiplier': [0.25],

    'prob_recover_gamma_shape': [7],
    'prob_recover_gamma_scale': [3],
    'prob_recover_gamma_loc': [0],
    'prob_recover_gamma_magnitude_multiplier': [0.75],

    'prob_virus_kill_host_gamma_shape': [5.2],
    'prob_virus_kill_host_gamma_scale': [3.2],
    'prob_virus_kill_host_gamma_loc': [0],
    'prob_virus_kill_host_gamma_magnitude_multiplier': [0.069],

    'prob_infectious_no_to_mild_symptom_gamma_shape': [4.1],
    'prob_infectious_no_to_mild_symptom_gamma_scale': [1],
    'prob_infectious_no_to_mild_symptom_gamma_loc': [0],
    'prob_infectious_no_to_mild_symptom_gamma_magnitude_multiplier': [0.75],

    'prob_infectious_no_to_severe_symptom_gamma_shape': [1],
    'prob_infectious_no_to_severe_symptom_gamma_scale': [2],
    'prob_infectious_no_to_severe_symptom_gamma_loc': [0],
    'prob_infectious_no_to_severe_symptom_gamma_magnitude_multiplier': [0.1],

    'prob_infectious_no_to_critical_symptom_gamma_shape': [1],
    'prob_infectious_no_to_critical_symptom_gamma_scale': [2.8],
    'prob_infectious_no_to_critical_symptom_gamma_loc': [0],
    'prob_infectious_no_to_critical_symptom_gamma_magnitude_multiplier': [0.15],

    'prob_infectious_mild_to_no_symptom_gamma_shape': [3],
    'prob_infectious_mild_to_no_symptom_gamma_scale': [3],
    'prob_infectious_mild_to_no_symptom_gamma_loc': [0],
    'prob_infectious_mild_to_no_symptom_gamma_magnitude_multiplier': [0.25],

    'prob_infectious_mild_to_severe_symptom_gamma_shape': [4.9],
    'prob_infectious_mild_to_severe_symptom_gamma_scale': [2.2],
    'prob_infectious_mild_to_severe_symptom_gamma_loc': [0],
    'prob_infectious_mild_to_severe_symptom_gamma_magnitude_multiplier': [0.11],

    'prob_infectious_mild_to_critical_symptom_gamma_shape': [3.3],
    'prob_infectious_mild_to_critical_symptom_gamma_scale': [3.1],
    'prob_infectious_mild_to_critical_symptom_gamma_loc': [0],
    'prob_infectious_mild_to_critical_symptom_gamma_magnitude_multiplier': [0.11],

    'prob_infectious_severe_to_no_symptom_gamma_shape': [3],
    'prob_infectious_severe_to_no_symptom_gamma_scale': [2],
    'prob_infectious_severe_to_no_symptom_gamma_loc': [0],
    'prob_infectious_severe_to_no_symptom_gamma_magnitude_multiplier': [0.001],

    'prob_infectious_severe_to_mild_symptom_gamma_shape': [5],
    'prob_infectious_severe_to_mild_symptom_gamma_scale': [3],
    'prob_infectious_severe_to_mild_symptom_gamma_loc': [0],
    'prob_infectious_severe_to_mild_symptom_gamma_magnitude_multiplier': [0.001],

    'prob_infectious_severe_to_critical_symptom_gamma_shape': [7],
    'prob_infectious_severe_to_critical_symptom_gamma_scale': [3],
    'prob_infectious_severe_to_critical_symptom_gamma_loc': [0],
    'prob_infectious_severe_to_critical_symptom_gamma_magnitude_multiplier': [0.01],

    'prob_infectious_critical_to_no_symptom_gamma_shape': [7],
    'prob_infectious_critical_to_no_symptom_gamma_scale': [1],
    'prob_infectious_critical_to_no_symptom_gamma_loc': [0],
    'prob_infectious_critical_to_no_symptom_gamma_magnitude_multiplier': [0.001],

    'prob_infectious_critical_to_mild_symptom_gamma_shape': [4],
    'prob_infectious_critical_to_mild_symptom_gamma_scale': [2],
    'prob_infectious_critical_to_mild_symptom_gamma_loc': [0],
    'prob_infectious_critical_to_mild_symptom_gamma_magnitude_multiplier': [0.001],

    'prob_infectious_critical_to_severe_symptom_gamma_shape': [5],
    'prob_infectious_critical_to_severe_symptom_gamma_scale': [2],
    'prob_infectious_critical_to_severe_symptom_gamma_loc': [0],
    'prob_infectious_critical_to_severe_symptom_gamma_magnitude_multiplier': [0.25],

    'prob_recovered_no_to_mild_complication': [0.016],
    'prob_recovered_no_to_severe_complication': [0],
    'prob_recovered_mild_to_no_complication': [0.02],
    'prob_recovered_mild_to_severe_complication': [0.02],
    'prob_recovered_severe_to_no_complication': [0.001],
    'prob_recovered_severe_to_mild_complication': [0.001],
    'prob_gain_immunity': [0.005],

    'hospital_bed_capacity_as_percent_of_population': [0.10],
    'hospital_bed_cost_per_day': [2000],
    'icu_bed_capacity_as_percent_of_population': [0.10],
    'icu_bed_cost_per_day': [3000],
    'ventilator_capacity_as_percent_of_population': [0.1],
    'ventilator_cost_per_day': [100],
    'drugX_capacity_as_percent_of_population': [0.1],
    'drugX_cost_per_day': [20],
}

start_date = datetime.datetime(2020, 2, 20) # Setting
num_iterations = 1 # Setting
num_max_steps_in_reality = 95 # Setting
num_max_steps_in_simulation = 165 # Setting
end_date_in_reality = start_date + datetime.timedelta(days=num_max_steps_in_reality) # 2020-05-25
end_date_in_simulation = start_date + datetime.timedelta(days=num_max_steps_in_simulation) # 2020-09-22 if num_max_steps_in_simulation == 215

try:
    br = BatchRunnerMP(BatchHostNetwork,
                     br_params,
                     iterations=num_iterations,
                     max_steps=num_max_steps_in_simulation,
                     model_reporters={'Data Collector': lambda m: m.datacollector})
except Exception as e:
    print('Multiprocessing batch run not applied, reason as:', e)
    br = CustomBatchRunner(BatchHostNetwork,
                     br_params,
                     iterations=num_iterations,
                     max_steps=num_max_steps_in_simulation,
                     model_reporters={'Data Collector': lambda m: m.datacollector})

def main(on_switch=False, graph_switch=False, stats_test_switch=False, save_switch=False,
         realworld_prediction_switch=False, filename_tag=''):
    if on_switch:
        br.run_all()
        br_df = br.get_model_vars_dataframe()
        br_step_data = pd.DataFrame()

        for i in range(len(br_df['Data Collector'])):
            if isinstance(br_df['Data Collector'][i], DataCollector):
                print('>>>>> Run #{}'.format(i))

                i_run_data = br_df['Data Collector'][i].get_model_vars_dataframe()
                i_run_data['Date'] = i_run_data.apply(lambda row: convert_time_to_date(row, 'Time', start_date), axis=1)

                br_step_data = br_step_data.append(i_run_data, ignore_index=True)
                model_param = i_run_data['Model params'][0]

                df_real = prepare_realworld_data().copy()
                df_real['date_formatted'] = pd.to_datetime(df_real['date_formatted'])
                df_real.sort_values(by=['date_formatted'])

                df_sim = i_run_data.copy()
                df_sim['Date'] = pd.to_datetime(df_sim['Date'])
                df_sim.sort_values(by=['Date'])

                df_merged = pd.merge(df_real, df_sim, how='outer', left_on=['date_formatted'],
                                     right_on=['Date'])

                if graph_switch:
                    print('>> For graphs')
                    print('Model param:', model_param)
                    graphing(df=df_merged)

                if stats_test_switch:
                    print('>> For statistical tests')
                    print('Model param:', model_param)
                    df_merged_sliced = df_merged[(df_merged['date_formatted'] >= start_date)
                                          & (df_merged['date_formatted'] <= end_date_in_reality)]
                    statistical_test_validation(df=df_merged_sliced)

                if realworld_prediction_switch:
                    print('>> For real-world predictions')
                    print('Model param:', model_param)

                    df_merged = predict_by_percent_change_of_another_col(
                        df=df_merged,
                        predicted_col='cumulative_cases',
                        feature_col='Cumulative test-confirmed infectious'
                    )
                    df_merged = predict_by_percent_change_of_another_col(
                        df=df_merged,
                        predicted_col='cumulative_deaths',
                        feature_col='Cumulative test-confirmed dead'
                    )
                    df_merged = predict_by_percent_change_of_another_col(
                        df=df_merged,
                        predicted_col='active_cases',
                        feature_col='Test-confirmed infectious'
                    )

            br_step_data['File ID'] = filename_tag

            if save_switch:
                br_step_data.to_csv(os.getcwd() +
                                    '\\project_result\\disease_model_step_data{}_p{}.csv'.format(filename_tag, i),
                                    index=False)
                df_merged.to_csv(os.getcwd() +
                                    '\\project_result\\disease_model_merged_data{}_p{}.csv'.format(filename_tag, i),
                                    index=False)

# Helper functions
curr_dir = os.getcwd()
covid19_dir = '\\data\Covid19Canada'
covid19_timeseries_prov_dir = covid19_dir+'\\timeseries_prov'
cases_timeseries_filename = 'cases_timeseries_prov.csv'
mortality_timeseries_filename = 'mortality_timeseries_prov.csv'
overall_timeseries_filename = 'active_timeseries_prov.csv'
testing_timeseries_filename = 'testing_timeseries_prov.csv'

project_result_dir = '\\project_result'
output_real_data_filename = 'realworldCovid19_step_data_processed.csv'
popn_factor = 1000000 # Setting

def convert_time_to_date(row, var, start_date):
    current_date = start_date + datetime.timedelta(days=(int(row[var]-1)))
    return current_date

def get_realworld_data():
    path_overall = curr_dir+covid19_timeseries_prov_dir+'\\'+overall_timeseries_filename
    path_testing = curr_dir+covid19_timeseries_prov_dir+'\\'+testing_timeseries_filename
    df_overall = pd.read_csv(path_overall, encoding='utf-8', low_memory=False)
    df_overall.rename(columns={'date_active': 'date'}, inplace=True)
    df_testing = pd.read_csv(path_testing, encoding='utf-8', low_memory=False)
    df_testing.rename(columns={'date_testing': 'date'}, inplace=True)
    df_merged = pd.merge(df_overall, df_testing, on=['province', 'date'], how='outer')
    df_merged['testing'].fillna(0, inplace=True)
    df_merged['cumulative_testing'].fillna(0, inplace=True)
    del df_merged['testing_info']
    return df_merged

def prepare_realworld_data():
    df_canada = get_realworld_data().copy()

    # Restrict location
    prov = 'Alberta'
    if prov == 'Alberta':
        prov_popn = 4.41 * 1000000 # Source: https://economicdashboard.alberta.ca/Population
    df = df_canada[df_canada['province'] == 'Alberta']

    # Restrict date range
    df['date_formatted'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df = df[(df['date_formatted'] >= start_date) & (df['date_formatted'] <= end_date_in_reality)]

    # Additional calculations
    df['total_n'] = prov_popn
    df['rate_per_1M_cumulative_test_done'] = df.apply(get_proportion,
                                                 numerator='cumulative_testing',
                                                 denominator='total_n',
                                                 multiplier = popn_factor,
                                                 axis=1)
    df['rate_per_1M_cumulative_test-confirmed_infectious'] = df.apply(get_proportion,
                                                 numerator='cumulative_cases',
                                                 denominator='total_n',
                                                 multiplier = popn_factor,
                                                 axis=1)
    df['rate_per_1M_cumulative_test-confirmed_dead'] = df.apply(get_proportion,
                                                 numerator='cumulative_deaths',
                                                 denominator='total_n',
                                                 multiplier = popn_factor,
                                                 axis=1)
    return df

def graphing(df):
    display_vars_for_df_real = ['rate_per_1M_cumulative_test_done',
                                'rate_per_1M_cumulative_test-confirmed_infectious',
                                'rate_per_1M_cumulative_test-confirmed_dead',
                                ]

    display_vars_for_df_sim = [
        'Rate per 1M cumulative test done',
        'Rate per 1M cumulative infectious',
        'Rate per 1M cumulative test-confirmed infectious',
        'Rate per 1M cumulative dead',
        'Rate per 1M cumulative test-confirmed dead',
    ]

    for var in display_vars_for_df_real:
        sns.lineplot(x='date_formatted', y=var, data=df)
        plt.xticks(rotation=15)
        plt.title('Title: Real-world '+var)
        plt.show()

    for var in display_vars_for_df_sim:
        sns.lineplot(x='Date', y=var, data=df)
        plt.xticks(rotation=15)
        plt.title('Title: Simulated '+var)
        plt.show()

def statistical_test_validation(df):
    maxlag = 10
    granger_test = 'ssr_ftest'  # options are 'params_ftest', 'ssr_ftest', 'ssr_chi2test', and 'lrtest'

    var_pair_p1 = ['rate_per_1M_cumulative_test_done',
                   'Rate per 1M cumulative test done']
    var_pair_p2 = ['rate_per_1M_cumulative_test-confirmed_infectious',
                   'Rate per 1M cumulative test-confirmed infectious']
    var_pair_p3 = ['rate_per_1M_cumulative_test-confirmed_dead',
                   'Rate per 1M cumulative test-confirmed dead']

    granger_test_result_p1 = grangercausalitytests(df[var_pair_p1], maxlag=maxlag, verbose=False)
    granger_test_result_p2 = grangercausalitytests(df[var_pair_p2], maxlag=maxlag, verbose=False)
    granger_test_result_p3 = grangercausalitytests(df[var_pair_p3], maxlag=maxlag, verbose=False)

    granger_p_values_p1 = [round(granger_test_result_p1[i + 1][0][granger_test][1], 4) for i in range(maxlag)]
    granger_min_p_value_p1 = np.min(granger_p_values_p1)
    granger_max_p_value_p1 = np.max(granger_p_values_p1)
    granger_mean_p_value_p1 = np.mean(granger_p_values_p1)

    granger_p_values_p2 = [round(granger_test_result_p2[i + 1][0][granger_test][1], 4) for i in range(maxlag)]
    granger_min_p_value_p2 = np.min(granger_p_values_p2)
    granger_max_p_value_p2 = np.max(granger_p_values_p2)
    granger_mean_p_value_p2 = np.mean(granger_p_values_p2)

    granger_p_values_p3 = [round(granger_test_result_p3[i + 1][0][granger_test][1], 4) for i in range(maxlag)]
    granger_min_p_value_p3 = np.min(granger_p_values_p3)
    granger_max_p_value_p3 = np.max(granger_p_values_p3)
    granger_mean_p_value_p3 = np.mean(granger_p_values_p3)

    print('p-value of {}: min={}, max={}, mean={}'.format(granger_test, granger_min_p_value_p1,
                                                          granger_max_p_value_p1, granger_mean_p_value_p1))
    print('p-value of {}: min={}, max={}, mean={}'.format(granger_test, granger_min_p_value_p2,
                                                          granger_max_p_value_p2, granger_mean_p_value_p2))
    print('p-value of {}: min={}, max={}, mean={}'.format(granger_test, granger_min_p_value_p3,
                                                          granger_max_p_value_p3, granger_mean_p_value_p3))

    pearson_r1, pearson_p1 = stats.pearsonr(df.dropna()[var_pair_p1[0]], df.dropna()[var_pair_p1[1]])
    pearson_r2, pearson_p2 = stats.pearsonr(df.dropna()[var_pair_p2[0]], df.dropna()[var_pair_p2[1]])
    pearson_r3, pearson_p3 = stats.pearsonr(df.dropna()[var_pair_p3[0]], df.dropna()[var_pair_p3[1]])

    print(f'Pearson r: {pearson_r1} and p-value: {pearson_p1}')
    print(f'Pearson r: {pearson_r2} and p-value: {pearson_p2}')
    print(f'Pearson r: {pearson_r3} and p-value: {pearson_p3}')

def predict_by_percent_change_of_another_col(df, predicted_col, feature_col):
    pct_chg_col = feature_col+' percent change'
    df[pct_chg_col] = (df[feature_col] - df[feature_col].shift(1).fillna(0))/\
        df[feature_col].shift(1).fillna(0)
    empty_cells = df[predicted_col].isna()
    percents = df[pct_chg_col].where(empty_cells, 0) + 1
    df[predicted_col] = df[predicted_col].ffill() * percents.cumprod()
    return df

def get_proportion(df, numerator, denominator, multiplier=1):
    try:
        return (df[numerator]/df[denominator])*multiplier
    except ZeroDivisionError:
        return math.inf

def pandas_output_setting():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 170)
    pd.options.mode.chained_assignment = None  # default='warn'

# Main function
if __name__ == '__main__':
    pandas_output_setting()
    main(on_switch=True, graph_switch=True, stats_test_switch=True, save_switch=False,
         realworld_prediction_switch=True, filename_tag='_vFinal')