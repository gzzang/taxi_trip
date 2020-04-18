# @Time    : 2020/3/25 20:38
# @Author  : gzzang
# @File    : main_echart
# @Project : taxi_trip

# 固定显示标签
# pyechart功能不全导致无法分别设置两类标签的属性
# 生成图形后需要手动调整label为true
# 2004182347

import pandas as pd
import numpy as np
import pickle as pk
import pdb

from pyecharts.charts import Geo
from pyecharts import options as opts
from pyecharts.globals import ChartType


def plot_geo(count_ar, cluster_centers, flag_car_type, flag_point_type, is_output_target, is_heatmap, is_show_all_point):
    c = Geo().add_schema(maptype="北京")
    if is_show_all_point:
        for i, value in enumerate(count_ar):
            c.add_coordinate(i+100, value[0], value[1])
        data_add = [(i+100, value[2]) for i, value in enumerate(count_ar)]
        legend_string = 'Termination' if flag_point_type else 'Initiation'

        if is_heatmap:
            c.add(legend_string, data_add, color='#030303', is_large=True, type_=ChartType.HEATMAP,)
            map_type = 'heat'
        else:
            c.add(legend_string, data_add, color='#030303', is_large=True, symbol_size=4,)
            map_type = 'point'

    for i, value in enumerate(cluster_centers):
        c.add_coordinate(i+1, value[0], value[1])
    if is_show_all_point:
        data_add = [(i+1, 0) for i, value in enumerate(cluster_centers)]
    else:
        data_add = [(i+1, 0) for i, value in enumerate(cluster_centers)]
    c.add('legend_string', data_add, color='black' ,symbol='diamond', symbol_size=10)

    # c.add_coordinate(34, cluster_centers[0][0], cluster_centers[0][1]+0.01)
    # c.add('', [(34,21)])

    c.set_series_opts(label_opts=opts.LabelOpts(is_show=not is_show_all_point, formatter='{b}', color='black',font_weight='bolder'))
    c.set_global_opts(visualmap_opts=[opts.VisualMapOpts(series_index=0, is_show=False, )],
                      title_opts=opts.TitleOpts(title="Taxi-Trip"),
                      legend_opts=opts.LegendOpts(is_show=False))

    if is_show_all_point:
        file_name ='out/echart/map_' + flag_point_type + '_' + flag_car_type + "_" + map_type + ".html"
    else:
        file_name = 'test.html'

    c.render(file_name)


def write(cluster_centers, flag_car_type, flag_point_type):
    pd.DataFrame(cluster_centers, columns=['lon', 'lat']).to_csv(
        'out/echart/result_' + flag_point_type + '_' + flag_car_type + ".csv", index=None)


with open('out/result_echart.pkl', 'rb') as f:
    result_echart = pk.load(f)

for flag_car_type in ['taxi', 'online']:
    for flag_point_type in ['arrival', 'departure']:
        for is_heatmap in [True, False]:
            plot_geo(count_ar=result_echart[(flag_car_type, flag_point_type)]['count_ar'],
                     cluster_centers=result_echart[(flag_car_type, flag_point_type)]['cluster_centers'],
                     flag_car_type=flag_car_type,
                     flag_point_type=flag_point_type,
                     is_output_target=False,
                     is_heatmap=is_heatmap,
                     is_show_all_point=True)
        write(cluster_centers=result_echart[(flag_car_type, flag_point_type)]['cluster_centers'],
              flag_car_type=flag_car_type,
              flag_point_type=flag_point_type)

# flag_car_type = 'taxi'
# flag_point_type = 'arrival'
# plot_geo(count_ar=result_echart[(flag_car_type, flag_point_type)]['count_ar'],
#                      cluster_centers=result_echart[(flag_car_type, flag_point_type)]['cluster_centers'],
#                      flag_car_type=flag_car_type,
#                      flag_point_type=flag_point_type,
#                      is_output_target=False,
#                      is_heatmap=is_heatmap,
#                      is_show_all_point=False)