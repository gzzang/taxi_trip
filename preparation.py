# @Time    : 2020/3/30 18:01
# @Author  : gzzang
# @File    : preparation
# @Project : taxi_trip

# 1.可以一次运行输出所有画图和表格数据
# 2.也可以按照需要输出

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle as pk

import time

import os
import pdb


def read_df_list(flag_car_type):
    return [read_day_df(day + 1, flag_car_type) for day in range(31)]


def read_day_df(day, flag_car_type):
    if flag_car_type == 'taxi':
        file_name = f'201803{day:02d}.csv'
    elif flag_car_type == 'online':
        file_name = f'WYC_OD_201803{day:02d}'
    file_path = 'data/' + flag_car_type + '/' + file_name
    return read_standard_df(file_path)


def read_standard_df(file_path):
    full_df = pd.read_csv(file_path, header=None)
    full_df.columns = ['id', 'date', 'hour', 'init', 'term', 'a', 'b']

    init_value = full_df['init'].to_numpy()
    init_lon, init_lat = convert_coordinate(init_value)

    full_df['init_lon'] = init_lon
    full_df['init_lat'] = init_lat

    term_value = full_df['term'].to_numpy()
    term_lon, term_lat = convert_coordinate(term_value)

    full_df['term_lon'] = term_lon
    full_df['term_lat'] = term_lat

    nonzero_df = full_df[(full_df['term'] != 0) & (full_df['init'] != 0)]
    standard_df = drop_noisy(nonzero_df, ['init_lat', 'init_lon', 'term_lat', 'term_lon'])
    return standard_df.drop(['init', 'term'], axis=1)


def drop_noisy(df, columns_str):
    df_describe = df.describe()
    for column in columns_str:
        mean = df_describe.loc['mean', column]
        std = df_describe.loc['std', column]
        minvalue = mean - 4 * std
        maxvalue = mean + 4 * std
    return df[(df[column] >= minvalue) & (df[column] <= maxvalue)]


def convert_coordinate(long_value):
    return (long_value // 100000) / 1000, (long_value % 100000) / 1000


def cal_result_echart(df_list_dict):
    result_echart = {}
    for flag_car_type in ['taxi', 'online']:
        df = pd.concat(df_list_dict[flag_car_type], axis=0, ignore_index=True)
        for flag_point_type in ['init', 'term']:
            print('--------')
            print(f'flag_car_type{flag_car_type}')
            print(f'flag_point_type{flag_point_type}')
            result_echart[(flag_car_type, flag_point_type)] = cal_car_type_result_echart_by_point_type(df,
                                                                                                       flag_point_type=flag_point_type,
                                                                                                       is_output_target=True)
    return result_echart


def cal_car_type_result_echart_by_point_type(df, flag_point_type, is_output_target):
    coordinate_df = cal_coordinate_df(df, flag_point_type=flag_point_type, is_output_target=is_output_target)
    count_df = coordinate_df.groupby(["lon", "lat"]).size().reset_index(name="time")
    count_ar = count_df.to_numpy()
    kmeans = KMeans(n_clusters=10, random_state=0).fit(coordinate_df)
    cluster_centers = kmeans.cluster_centers_
    return {'count_ar': count_ar, 'cluster_centers': cluster_centers}


def cal_coordinate_df(df, flag_point_type, is_output_target):
    selected_df = pick_selected_df(df, flag_point_type=flag_point_type)
    if is_output_target:
        column = flag_point_type
    else:
        if flag_point_type == 'init':
            column = 'term'
        elif flag_point_type == 'term':
            column = 'init'
    coordinate_df = selected_df[[column + "_lon", column + "_lat"]]
    coordinate_df.columns = ['lon', 'lat']
    return coordinate_df


def cal_result_figure(car_dict_day_list_df):
    month_day_number = 31
    week_day_number = 7
    day_hour_number = 24

    car_dict_point_dict_day_hour_flow = {}
    for car_type, day_list_df in car_dict_day_list_df.items():
        point_dict_day_hour_flow = {}
        for point_type in ['init', 'term']:
            day_list_hour_flow = []
            for df in day_list_df:
                selected_df = pick_selected_df(df, point_type)

                hour_count = selected_df['hour'].value_counts()
                hour_count_index = hour_count.index.to_numpy()
                hour_count_value = hour_count.to_numpy()
                hour_flow = np.zeros(day_hour_number, dtype=int)
                hour_flow[hour_count_index] = hour_count_value

                day_list_hour_flow.append(hour_flow)
            point_dict_day_hour_flow[point_type] = np.array(day_list_hour_flow)
        car_dict_point_dict_day_hour_flow[car_type] = point_dict_day_hour_flow

    return {'month_day_number': month_day_number,
            'day_hour_number': day_hour_number,
            'car_dict_point_dict_day_hour_flow': car_dict_point_dict_day_hour_flow,
            'week_day_number': week_day_number
            }


def pick_selected_df(df, flag_point_type):
    airport_min_lon = 116.573599
    airport_max_lon = 116.626814
    airport_min_lat = 40.047361
    airport_max_lat = 40.108833

    if flag_point_type == 'init':
        column = 'term'
    elif flag_point_type == 'term':
        column = 'init'
    return df[(df[column + '_lon'] >= airport_min_lon) & (df[column + '_lon'] <= airport_max_lon) &
              (df[column + '_lat'] >= airport_min_lat) & (df[column + '_lat'] <= airport_max_lat)]


if __name__ == '__main__':
    # 1 需要绘制流量图
    # 2 需要绘制区域统计和聚类中心

    temp = time.time()

    df_list_dict = {}
    for flag_car_type in ['taxi', 'online']:
        print('----read_df_list----')
        print(f'flag_car_type:{flag_car_type}')
        df_list_dict[flag_car_type] = read_df_list(flag_car_type)

    temp2 = time.time()

    # result_echart = cal_result_echart(df_list_dict)
    # with open('out/result_echart.pkl', 'wb') as f:
    #     pk.dump(result_echart, f)

    temp3 = time.time()

    result_figure = cal_result_figure(df_list_dict)
    with open('out/result_figure.pkl', 'wb') as f:
        pk.dump(result_figure, f)

    print(temp2 - temp)
    print(temp3 - temp2)
    print(time.time() - temp3)
