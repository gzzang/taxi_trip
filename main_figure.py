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

with open('out/result_figure.pkl', 'rb') as f:
    result_figure = pk.load(f)

output_path = 'out/figure/'

month_day_number = result_figure['month_day_number']
day_hour_number = result_figure['day_hour_number']
week_day_number = result_figure['week_day_number']
car_dict_point_dict_day_hour_flow = result_figure['car_dict_point_dict_day_hour_flow']
day_list_point_dict_hour_flow = result_figure['day_list_point_dict_hour_flow']
point_list_string = result_figure['point_list_string']
day_list_dict_hour_flow = result_figure['day_list_dict_hour_flow']
point_dict_day_flow = result_figure['point_dict_day_flow']
df_day_flow = result_figure['df_day_flow']
week = result_figure['week']
df_week_day_flow = result_figure['df_week_day_flow']
week_string = result_figure['week_string']
average_week_day_flow_df = result_figure['average_week_day_flow_df']
week_number = result_figure['week_number']
week_list = result_figure['week_list']
point_dict_week_day_average_flow = result_figure['point_dict_week_day_average_flow']

# 第一系列图
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
for i, flow in enumerate(day_list_dict_hour_flow):
    pd.DataFrame(flow).to_csv(output_path + f'table_flow_by_hours_{i + 1}.csv', index=None)

# 第二系列图
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
df_day_flow.to_csv(output_path + 'table_month_flow_by_days.csv', index=None)

# 第三系列图
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
df_week_day_flow.to_csv(output_path + 'table_week_flow_by_days.csv', index=None)

# 第四系列图
plt.figure()
for key, value in point_dict_week_day_average_flow.items():
    plt.plot(range(week_day_number), value, label=point_list_string[key])
plt.xticks(range(week_day_number), week_string)
plt.ylabel(u'数量')
plt.legend()
figure_name = 'average_week_flow_by_days'
plt.gcf().canvas.set_window_title(figure_name)
plt.savefig(output_path + 'figure_' + figure_name + '.png')
plt.close()

# 第四系列表
average_week_day_flow_df.to_csv(output_path + 'table_average_week_flow_by_days.csv', index=None)

# 第五系列图
for string in ['init', 'term']:
    plt.figure()
    for j in range(week_number):
        plt.plot(range(week_day_number), point_dict_day_flow[string][week_list[j]], label=f'第{j + 1}周')
    plt.xticks(range(week_day_number), week_string)
    plt.ylabel(u'数量')
    plt.legend()
    figure_name = string + '_week_flow'
    plt.gcf().canvas.set_window_title(figure_name)
    plt.savefig(output_path + 'figure_' + figure_name + '.png')
    plt.close()
