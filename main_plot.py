# @Time    : 2020/3/13 18:47
# @Author  : gzzang
# @File    : main_plot
# @Project : taxi_trip


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def cal_coordinate(long_value):
    return np.floor_divide(long_value, 100000) / 1000, np.mod(long_value, 100000) / 1000


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

airport_min_longitude = 116.573599
airport_max_longitude = 116.626814
airport_min_latitude = 40.047361
airport_max_latitude = 40.108833

month_day_number = 31
week_day_number = 7
week_first_day = 12
day_hour_number = 24

column_list = ['id', 'date', 'hour', 'init', 'term', 'a', 'b']

car_type_list = ['taxi', 'online']
type_hour_init_flow = np.zeros([2, month_day_number, day_hour_number], dtype=int)
type_hour_term_flow = np.zeros([2, month_day_number, day_hour_number], dtype=int)
day_init_flow = np.zeros([2, month_day_number], dtype=int)
day_term_flow = np.zeros([2, month_day_number], dtype=int)
for j, car_type in enumerate(car_type_list):
    data_folder_path = r'data/' + car_type + '/'
    file_name_list = os.listdir(data_folder_path)

    day_number = len(file_name_list)
    init_flow = np.zeros(day_number)
    term_flow = np.zeros(day_number)

    day_hour_init_flow = {}
    day_hour_term_flow = {}
    for i, one_file_name in enumerate(file_name_list):
        df = pd.read_csv(data_folder_path + one_file_name, header=None)
        df.columns = column_list

        init_lon, init_lat = cal_coordinate(df['init'].to_numpy())
        term_lon, term_lat = cal_coordinate(df['term'].to_numpy())

        df['init_lon'] = init_lon
        df['init_lat'] = init_lat
        df['term_lon'] = term_lon
        df['term_lat'] = term_lat

        init_flag = (init_lon > airport_min_longitude) & (init_lon < airport_max_longitude) & (
                init_lat > airport_min_latitude) & (init_lat < airport_max_latitude)
        term_flag = (term_lon > airport_min_longitude) & (term_lon < airport_max_longitude) & (
                term_lat > airport_min_latitude) & (term_lat < airport_max_latitude)

        day_init_flow[j][i] = int(init_flag.sum())
        day_term_flow[j][i] = int(term_flag.sum())

        df['init_flag'] = init_flag
        df['term_flag'] = term_flag

        hour_init_flow = pd.pivot_table(df, index=['hour'], values=['init_flag'], aggfunc='sum').to_numpy()
        hour_term_flow = pd.pivot_table(df, index=['hour'], values=['term_flag'], aggfunc='sum').to_numpy()
        type_hour_init_flow[j, i, :] = hour_init_flow.flatten()
        type_hour_term_flow[j, i, :] = hour_term_flow.flatten()

output_path = 'output/'

for i, one_file_name in enumerate(file_name_list):
    plt.figure()
    xtick = np.arange(24)

    plt.plot(xtick, type_hour_init_flow.sum(axis=0)[i, :])
    plt.plot(xtick, type_hour_term_flow.sum(axis=0)[i, :])
    plt.xticks(xtick)
    plt.xlabel(u'小时')
    plt.ylabel(u'数量')
    plt.legend([u'出发', u'到达'])
    figure_name = f'flow_by_hours_{i + 1}'
    plt.gcf().canvas.set_window_title(figure_name)
    plt.savefig(output_path + 'figure_' + figure_name + '.png')
    plt.close()

plt.figure()
plt.plot(np.sum(day_init_flow, axis=0))
plt.plot(np.sum(day_term_flow, axis=0))
plt.xlabel(u'日期')
plt.ylabel(u'数量')
plt.legend([u'出发', u'到达'])
figure_name = 'month_flow_by_days'
plt.gcf().canvas.set_window_title(figure_name)
plt.savefig(output_path + 'figure_' + figure_name + '.png')
plt.close()

week_init_flow = np.sum(day_init_flow, axis=0)[(week_first_day - 1):(week_first_day + week_day_number - 1)]
week_term_flow = np.sum(day_term_flow, axis=0)[(week_first_day - 1):(week_first_day + week_day_number - 1)]

plt.figure()
plt.plot(np.arange(week_day_number) + week_first_day, week_init_flow)
plt.plot(np.arange(week_day_number) + week_first_day, week_term_flow)
plt.xlabel(u'日期')
plt.ylabel(u'数量')
plt.legend([u'出发', u'到达'])
figure_name = 'week_flow_by_days'
plt.gcf().canvas.set_window_title(figure_name)
plt.savefig(output_path + 'figure_' + figure_name + '.png')
plt.close()

table_column_list = ['taxi_init', 'taxi_term', 'online_init', 'online_term', 'total_init', 'total_term']

for i, one_file_name in enumerate(file_name_list):
    one_type_hour_init_flow = type_hour_init_flow[:, i, :]
    one_type_hour_term_flow = type_hour_term_flow[:, i, :]
    hour_flow = np.vstack((one_type_hour_init_flow[0, :], one_type_hour_term_flow[0, :], one_type_hour_init_flow[1, :],
                           one_type_hour_term_flow[1, :], np.sum(one_type_hour_init_flow, axis=0),
                           np.sum(one_type_hour_term_flow, axis=0)))
    df_hour_flow = pd.DataFrame(hour_flow.T)
    df_hour_flow.columns = table_column_list
    df_hour_flow.to_csv(output_path + f'table_flow_by_hours_{i + 1}.csv', index=None)

day_flow = np.vstack((day_init_flow[0, :], day_term_flow[0, :], day_init_flow[1, :], day_term_flow[1, :],
                      np.sum(day_init_flow, axis=0), np.sum(day_term_flow, axis=0)))
df_day_flow = pd.DataFrame(day_flow.T)
df_day_flow.columns = table_column_list
df_day_flow.to_csv(output_path + 'table_month_flow_by_days.csv', index=None)

df_day_flow = pd.DataFrame(day_flow[:, (week_first_day - 1):(week_first_day + week_day_number - 1)].T)
df_day_flow.columns = table_column_list
df_day_flow.to_csv(output_path + 'table_week_flow_by_days.csv', index=None)
