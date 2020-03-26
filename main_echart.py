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


def cal_coordinate_count_of_one_day(file_path, initiation_flag):
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

    if initiation_flag:
        airport_init_df = get_airport(standard_df, 'init')
        count_airport_init_df = airport_init_df.groupby(["term_longitude", "term_latitude"]).size().reset_index(
            name="time")
        # count_airport_init_ar = count_airport_init_df.to_numpy()
        return count_airport_init_df
    else:
        airport_term_df = get_airport(standard_df, 'term')
        count_airport_term_df = airport_term_df.groupby(["init_longitude", "init_latitude"]).size().reset_index(
            name="time")
        # count_airport_term_ar = count_airport_term_df.to_numpy()
        return count_airport_term_df


car_type_list = ['taxi', 'online']
for car_type in car_type_list:
    print(f'car_type:{car_type}')
    coordinate_list = ['init', 'term']
    for bool in [True, False]:
        airport_coordinate_type = coordinate_list[not bool]
        another_coordinate_type = coordinate_list[bool]
        print(f'airport_coordinate_type:{airport_coordinate_type}')
        data_folder_path = r'data/' + car_type + '/'
        file_name_list = os.listdir(data_folder_path)
        result = pd.DataFrame()
        for one_file_name in file_name_list:
            print('.')
            file_path = data_folder_path + one_file_name
            initiation_flag = bool
            count_airport_init_df = cal_coordinate_count_of_one_day(file_path, initiation_flag=initiation_flag)
            result = result.append(count_airport_init_df)

        count_airport_init_df = \
        result.groupby([another_coordinate_type + "_longitude", another_coordinate_type + "_latitude"])[
            'time'].sum().reset_index(name="time")

        count_airport_init_ar = count_airport_init_df.to_numpy()
        map_type_list = ['point', 'heat']
        for map_type in map_type_list:
            c = Geo().add_schema(maptype="北京")
            for i, value in enumerate(count_airport_init_ar):
                c.add_coordinate(i, value[0], value[1])

            data_add = [(i, value[2]) for i, value in enumerate(count_airport_init_ar)]

            legend_string = 'Termination' if bool else 'Initiation'

            if map_type == 'point':
                c.add(legend_string, data_add, is_large=True, symbol_size=4)
            elif map_type == 'heat':
                c.add(legend_string, data_add, is_large=True, type_=ChartType.HEATMAP, )
            c.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            c.set_global_opts(
                visualmap_opts=opts.VisualMapOpts(type_="color", min_=0, max_=count_airport_init_ar[:, 2].max(),
                                                  is_piecewise=True),
                title_opts=opts.TitleOpts(title="Taxi-Trip"), legend_opts=opts.LegendOpts(is_show=False))
            c.render("map_airport_"+car_type+'_'+airport_coordinate_type+ "_" + map_type + ".html")

