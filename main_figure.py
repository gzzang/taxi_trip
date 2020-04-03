# @Time    : 2020/3/13 18:47
# @Author  : gzzang
# @File    : main_plot
# @Project : taxi_trip


import pandas as pd
import numpy as np
import os
import pickle as pk
import matplotlib.pyplot as plt
import pdb

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

with open('out/result_figure.pkl', 'rb') as point_dict_week_day_average_flow:
    result_figure = pk.load(point_dict_week_day_average_flow)

output_path = 'out/figure/'

month_day_number = result_figure['month_day_number']
day_hour_number = result_figure['day_hour_number']
week_day_number = result_figure['week_day_number']
car_dict_point_dict_day_hour_flow = result_figure['car_dict_point_dict_day_hour_flow']

week_first_day = 12

car_type_list = ['taxi', 'online']
point_type_list = ['init', 'term']
point_list_string = {'init': u'到达机场', 'term': u'离开机场'}

car_dict_point_dict_day_flow = {
    car_type: {point_type: day_hour_flow.sum(axis=1) for point_type, day_hour_flow in point_dict_day_hour_flow.items()}
    for car_type, point_dict_day_hour_flow in car_dict_point_dict_day_hour_flow.items()}

# 第一系列图
day_list_point_dict_hour_flow = [{point_type: np.array([car_dict_point_dict_day_hour_flow[car_type][point_type][i, :]
                                                        for car_type in car_type_list]).sum(axis=0)
                                  for point_type in point_type_list}
                                 for i in range(month_day_number)]

for i, flow_dict in enumerate(day_list_point_dict_hour_flow):
    plt.figure()
    xtick = np.arange(day_hour_number)
    for key, value in flow_dict.items():
        plt.plot(xtick, value, label=point_list_string[key])
    plt.xticks(xtick)
    plt.xlabel(u'小时')
    plt.ylabel(u'数量')
    plt.legend()
    figure_name = f'flow_by_hours_{i + 1}'
    plt.gcf().canvas.set_window_title(figure_name)
    plt.savefig(output_path + 'figure_' + figure_name + '.png')
    plt.close()

# 第一系列表
car_plus_dict_point_dict_day_hour_flow = car_dict_point_dict_day_hour_flow.copy()

car_plus_dict_point_dict_day_hour_flow['total'] = {'init': car_plus_dict_point_dict_day_hour_flow['taxi']['init'] + \
                                                           car_plus_dict_point_dict_day_hour_flow['online']['init'],
                                                   'term': car_plus_dict_point_dict_day_hour_flow['taxi']['term'] + \
                                                           car_plus_dict_point_dict_day_hour_flow['online']['term']}

day_list_dict_hour_flow = [{car_type + '_' + point_type: day_flow[i, :]
                            for point_type, car_dict_day_hour_flow in car_plus_dict_point_dict_day_hour_flow.items()
                            for car_type, day_flow in car_dict_day_hour_flow.items()}
                           for i in range(month_day_number)]

table_column_list = ['taxi_init', 'taxi_term', 'online_init', 'online_term', 'total_init', 'total_term']
for i, flow in enumerate(day_list_dict_hour_flow):
    pd.DataFrame(flow).to_csv(output_path + f'table_flow_by_hours_{i + 1}.csv', index=None)

# 第二系列图

point_dict_day_flow = {point_type: np.array(
    [point_dict_day_flow[point_type] for point_dict_day_flow in car_dict_point_dict_day_flow.values()]).sum(axis=0) for
                       point_type in point_type_list}

plt.figure()
for key, value in point_dict_day_flow.items():
    plt.plot([i + 1 for i in range(month_day_number)], value, label=point_list_string[key])
plt.xticks([i + 1 for i in range(month_day_number)])
plt.xlabel(u'日期')
plt.ylabel(u'数量')
plt.legend()
figure_name = 'month_flow_by_days'
plt.gcf().canvas.set_window_title(figure_name)
plt.savefig(output_path + 'figure_' + figure_name + '.png')
plt.close()

# 第二系列表
df_day_flow = pd.DataFrame({point_type + '_' + car_type: day_flow for car_type, point_dict_day_flow in
                            car_dict_point_dict_day_flow.items() for point_type, day_flow in
                            point_dict_day_flow.items()})
df_day_flow.to_csv(output_path + 'table_month_flow_by_days.csv', index=None)

# 第三系列图

week = range(week_first_day - 1, week_first_day + week_day_number - 1)

plt.figure()
for key, value in point_dict_day_flow.items():
    plt.plot([i + 1 for i in week], value[week], label=point_list_string[key])
plt.xlabel(u'日期')
plt.ylabel(u'数量')
plt.legend()
figure_name = 'week_flow_by_days'
plt.gcf().canvas.set_window_title(figure_name)
plt.savefig(output_path + 'figure_' + figure_name + '.png')
plt.close()

# 第三系列表

df_week_day_flow = df_day_flow[(week_first_day - 1):(week_first_day + week_day_number - 1)]
df_week_day_flow.to_csv(output_path + 'table_week_flow_by_days.csv', index=None)

# 第四系列图
point_type_day_flow = pd.DataFrame(point_dict_day_flow)
point_type_day_flow['week'] = (np.arange(month_day_number) + 3) % week_day_number
average_week_day_flow_df = point_type_day_flow.groupby(by='week').mean()
column_string_list = average_week_day_flow_df.columns.to_list()
average_week_day_flow = average_week_day_flow_df.to_numpy().T

point_dict_week_day_average_flow = {}
for key, value in zip(column_string_list, average_week_day_flow):
    point_dict_week_day_average_flow[key] = value

plt.figure()
for key, value in point_dict_week_day_average_flow.items():
    plt.plot([i + 1 for i in range(week_day_number)], value, label=point_list_string[key])
plt.xlabel(u'周')
plt.ylabel(u'数量')
plt.legend()
figure_name = 'average_week_flow_by_days'
plt.gcf().canvas.set_window_title(figure_name)
plt.savefig(output_path + 'figure_' + figure_name + '.png')
plt.close()

# 第四系列表
average_week_day_flow_df.to_csv(output_path + 'table_average_week_flow_by_days.csv', index=None)
