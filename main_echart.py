# @Time    : 2020/3/25 20:38
# @Author  : gzzang
# @File    : main_echart
# @Project : taxi_trip


import pandas as pd
import numpy as np
import os
import pdb

from pyecharts.charts import Geo
from pyecharts import options as opts
from pyecharts.globals import ChartType


def drop_noisy(df, columns_str):
    df_copy = df.copy()
    df_describe = df_copy.describe()
    for column in columns_str:
        mean = df_describe.loc['mean', column]
        std = df_describe.loc['std', column]
        minvalue = mean - 4 * std
        maxvalue = mean + 4 * std
        df_copy = df_copy[df_copy[column] >= minvalue]
        df_copy = df_copy[df_copy[column] <= maxvalue]
    return df_copy


def get_airport(df, column_str):
    airport_min_longitude = 116.573599
    airport_max_longitude = 116.626814
    airport_min_latitude = 40.047361
    airport_max_latitude = 40.108833

    df_copy = df.copy()
    df_copy = df_copy[df_copy[column_str + '_longitude'] >= airport_min_longitude]
    df_copy = df_copy[df_copy[column_str + '_longitude'] <= airport_max_longitude]
    df_copy = df_copy[df_copy[column_str + '_latitude'] >= airport_min_latitude]
    df_copy = df_copy[df_copy[column_str + '_latitude'] <= airport_max_latitude]
    return df_copy


def cal_coordinate_count_of_one_day(file_path, is_airport_initiation, is_output_airport=False):
    coordinate_list = ['init', 'term']
    airport_coordinate_type = coordinate_list[not is_airport_initiation]
    another_coordinate_type = coordinate_list[is_airport_initiation]

    full_df = pd.read_csv(file_path, header=None)
    full_df.columns = ['id', 'date', 'hour', 'init', 'term', 'a', 'b']

    init_value = full_df['init'].to_numpy()
    init_longitude = np.floor_divide(init_value, 100000) / 1000
    init_latitude = np.mod(init_value, 100000) / 1000

    full_df['init_longitude'] = init_longitude
    full_df['init_latitude'] = init_latitude

    term_value = full_df['term'].to_numpy()
    term_longitude = np.floor_divide(term_value, 100000) / 1000
    term_latitude = np.mod(term_value, 100000) / 1000

    full_df['term_longitude'] = term_longitude
    full_df['term_latitude'] = term_latitude

    nonzero_df = full_df[(full_df['term'] != 0) & (full_df['init'] != 0)]
    standard_df = drop_noisy(nonzero_df, ['init_latitude', 'init_longitude', 'term_latitude', 'term_longitude'])

    airport_init_df = get_airport(standard_df, airport_coordinate_type)
    if is_output_airport:
        count_point_df = airport_init_df.groupby(
            [airport_coordinate_type + "_longitude", airport_coordinate_type + "_latitude"]).size().reset_index(
            name="time")
    else:
        count_point_df = airport_init_df.groupby(
            [another_coordinate_type + "_longitude", another_coordinate_type + "_latitude"]).size().reset_index(
            name="time")
    count_point_df.columns = ['longitude', 'latitude', 'time']

    return count_point_df


def plot(is_car_taxi, is_airport_initiation, is_output_airport=False):
    car_type = 'taxi' if is_car_taxi else 'online'
    print(f'car_type:{car_type}')
    airport_type = 'init' if is_airport_initiation else 'term'
    print(f'airport_coordinate_type:{airport_type}')
    output_type = 'airport' if is_output_airport else 'other'
    print(f'output_type:{output_type}')

    data_folder_path = r'data/' + car_type + '/'
    file_name_list = os.listdir(data_folder_path)
    count_df = pd.DataFrame()
    for one_file_name in file_name_list:
        print('.', end='')
        file_path = data_folder_path + one_file_name
        count_df = count_df.append(
            cal_coordinate_count_of_one_day(file_path, is_airport_initiation=is_airport_initiation,
                                            is_output_airport=is_output_airport))
    print('')
    count_df = count_df.groupby(["longitude", "latitude"])['time'].sum().reset_index(name="time")
    count_ar = count_df.to_numpy()
    map_type_list = ['point', 'heat']
    for map_type in map_type_list:
        c = Geo().add_schema(maptype="北京")
        for i, value in enumerate(count_ar):
            c.add_coordinate(i, value[0], value[1])

        data_add = [(i, value[2]) for i, value in enumerate(count_ar)]

        legend_string = 'Termination' if is_airport_initiation else 'Initiation'

        if map_type == 'point':
            c.add(legend_string, data_add, is_large=True, symbol_size=4)
        elif map_type == 'heat':
            c.add(legend_string, data_add, is_large=True, type_=ChartType.HEATMAP, )
        c.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        c.set_global_opts(
            visualmap_opts=opts.VisualMapOpts(type_="color", min_=0, max_=count_ar[:, 2].max(),
                                              is_piecewise=True),
            title_opts=opts.TitleOpts(title="Taxi-Trip"), legend_opts=opts.LegendOpts(is_show=False))
        c.render('map_' + output_type + '_' + car_type + '_' + airport_type + "_" + map_type + ".html")


plot(is_car_taxi=True, is_airport_initiation=True, is_output_airport=False)

car_type_list = ['taxi', 'online']

for is_car_taxi in [False, True]:
    for is_airport_initiation in [False, True]:
        plot(is_car_taxi=is_car_taxi, is_airport_initiation=is_airport_initiation)
