# @Time    : 2020/3/25 20:38
# @Author  : gzzang
# @File    : main_echart
# @Project : taxi_trip


import pandas as pd
import numpy as np
import pickle as pk

from pyecharts.charts import Geo
from pyecharts import options as opts
from pyecharts.globals import ChartType


def plot_geo(count_ar, cluster_centers, flag_car_type, flag_point_type, is_output_target, is_heatmap):
    c = Geo().add_schema(maptype="北京")
    for i, value in enumerate(count_ar):
        c.add_coordinate(i, value[0], value[1])
    data_add = [(i, value[2]) for i, value in enumerate(count_ar)]
    legend_string = 'Termination' if flag_point_type else 'Initiation'

    if is_heatmap:
        c.add(legend_string, data_add, color='#030303', is_large=True, type_=ChartType.HEATMAP, )
        map_type = 'heat'
    else:
        c.add(legend_string, data_add, color='#030303', is_large=True, symbol_size=4, )
        map_type = 'point'

    for i, value in enumerate(cluster_centers):
        c.add_coordinate(-i - 1, value[0], value[1])
    data_add = [(-i - 1, 0) for i, value in enumerate(cluster_centers)]
    c.add('legend_string', data_add, color="black", symbol='diamond', symbol_size=10)

    c.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    c.set_global_opts(visualmap_opts=[opts.VisualMapOpts(series_index=0, is_show=False, )],
                      title_opts=opts.TitleOpts(title="Taxi-Trip"),
                      legend_opts=opts.LegendOpts(is_show=False))

    c.render('out/echart/map_' + flag_point_type + '_' + flag_car_type + "_" + map_type + ".html")


def write(cluster_centers, flag_car_type, flag_point_type):
    pd.DataFrame(cluster_centers, columns=['lon', 'lat']).to_csv(
        'out/echart/result_' + flag_point_type + '_' + flag_car_type + ".csv", index=None)


with open('out/result_echart.pkl', 'rb') as f:
    result_echart = pk.load(f)

for flag_car_type in ['taxi', 'online']:
    for flag_point_type in ['arrival', 'departure']:
        plot_geo(count_ar=result_echart[(flag_car_type, flag_point_type)]['count_ar'],
                 cluster_centers=result_echart[(flag_car_type, flag_point_type)]['cluster_centers'],
                 flag_car_type=flag_car_type,
                 flag_point_type=flag_point_type,
                 is_output_target=False,
                 is_heatmap=True)
        write(cluster_centers=result_echart[(flag_car_type, flag_point_type)]['cluster_centers'],
              flag_car_type=flag_car_type,
              flag_point_type=flag_point_type)
