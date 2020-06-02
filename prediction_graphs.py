import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

curr_dir = os.getcwd()
project_result_dir = '\\project_result'
full_project_result_dir = curr_dir+project_result_dir+'\\'
input_filename_core = 'disease_model_merged_data_vFinal_p'
output_filename_core = 'Graph'
start_prediction_date = datetime.datetime(2020, 5, 26) # Setting

def main(graph_switch=False, save_switch=False):
    pandas_output_setting()
    cols = {'Predicted cumulative count of infected cases in Alberta (Demo only)': 'cumulative_cases',
            'Predicted cumulative count of deaths in Alberta (Demo only)': 'cumulative_deaths'
            }
    df_master = get_merged_project_data()
    df_master['Date'] = pd.to_datetime(df_master['Date'])
    df_id_list = df_master['Run'].unique().tolist()

    if graph_switch:
        for key, col_label in cols.items():
            # Draw graphs
            fig, ax = plt.subplots()
            for df_id in df_id_list:
                df_master_sliced = df_master[df_master['Run'] == df_id]
                row_index = df_master_sliced[df_master_sliced['Date'] == \
                                             start_prediction_date].index.item()

                ax.plot(df_master_sliced['Date'][row_index:], df_master_sliced[col_label][row_index:])
                ax.plot(df_master_sliced['Date'][:row_index+1], df_master_sliced[col_label][:row_index+1])

            # Set formats and labels
            date_format = DateFormatter('%Y-%m')
            ax.xaxis.set_major_formatter(date_format)
            ax.xaxis.set_minor_formatter(date_format)
            ax.set_title('Title: {}'.format(key), fontsize=12)
            plt.xlabel('Time in Year and Month')
            plt.ylabel('Count')
            plt.figtext(0.99, 0.005, '(Simulated predictions started: {})'.format(start_prediction_date),
                        horizontalalignment='right')
            fig.autofmt_xdate()

            if save_switch:
                output_file_name = '{}{}_{}.png'.format(full_project_result_dir, output_filename_core, key)
                plt.savefig(output_file_name)

            plt.show()

### Helper functions ###
def pandas_output_setting():
    '''Set pandas output display setting'''
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 170)
    pd.options.mode.chained_assignment = None  # default='warn'

def get_merged_project_data():
    dfs_merged = None

    for filename in os.listdir(path=full_project_result_dir):
        if (input_filename_core in filename):
            path = full_project_result_dir+filename
            df = pd.read_csv(path, encoding='utf-8', low_memory=False)

            if dfs_merged is None:
                dfs_merged = df
            else:
                dfs_merged = pd.concat([dfs_merged, df])

    return dfs_merged

if __name__ == '__main__':
    main(graph_switch=True, save_switch=False)