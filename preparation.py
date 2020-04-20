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
        for flag_point_type in ['arrival', 'departure']:
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
        if flag_point_type == 'arrival':
            column = 'init'
        elif flag_point_type == 'departure':
            column = 'term'
    else:
        if flag_point_type == 'arrival':
            column = 'term'
        elif flag_point_type == 'departure':
            column = 'init'
    coordinate_df = selected_df[[column + "_lon", column + "_lat"]]
    coordinate_df.columns = ['lon', 'lat']
    return coordinate_df


def cal_coordinate_and_hour_df(df, flag_point_type, is_output_target):
    selected_df = pick_selected_df(df, flag_point_type=flag_point_type)
    if is_output_target:
        if flag_point_type == 'arrival':
            column = 'init'
        elif flag_point_type == 'departure':
            column = 'term'
    else:
        if flag_point_type == 'arrival':
            column = 'term'
        elif flag_point_type == 'departure':
            column = 'init'
    coordinate_df = selected_df[[column + "_lon", column + "_lat", 'hour']]
    coordinate_df.columns = ['lon', 'lat', 'hour']
    return coordinate_df


def cal_result_figure(car_dict_day_list_df):
    month_day_number = 31
    week_day_number = 7
    day_hour_number = 24

    car_dict_point_dict_day_hour_flow = {}
    for car_type, day_list_df in car_dict_day_list_df.items():
        point_dict_day_hour_flow = {}
        for point_type in ['arrival', 'departure']:
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

    week_first_day = 12

    car_type_list = ['taxi', 'online']
    point_type_list = ['arrival', 'departure']
    point_list_string = {'arrival': u'到达机场', 'departure': u'离开机场'}

    car_dict_point_dict_day_flow = {
        car_type: {point_type: day_hour_flow.sum(axis=1) for point_type, day_hour_flow in
                   point_dict_day_hour_flow.items()}
        for car_type, point_dict_day_hour_flow in car_dict_point_dict_day_hour_flow.items()}

    day_list_point_dict_hour_flow = [
        {point_type: np.array([car_dict_point_dict_day_hour_flow[car_type][point_type][i, :]
                               for car_type in car_type_list]).sum(axis=0)
         for point_type in point_type_list}
        for i in range(month_day_number)]

    car_plus_dict_point_dict_day_hour_flow = car_dict_point_dict_day_hour_flow.copy()

    car_plus_dict_point_dict_day_hour_flow['total'] = {
        'arrival': car_plus_dict_point_dict_day_hour_flow['taxi']['arrival'] + \
                   car_plus_dict_point_dict_day_hour_flow['online']['arrival'],
        'departure': car_plus_dict_point_dict_day_hour_flow['taxi']['departure'] + \
                     car_plus_dict_point_dict_day_hour_flow['online']['departure']}

    day_list_dict_hour_flow = [{car_type + '_' + point_type: day_flow[i, :]
                                for point_type, car_dict_day_hour_flow in car_plus_dict_point_dict_day_hour_flow.items()
                                for car_type, day_flow in car_dict_day_hour_flow.items()}
                               for i in range(month_day_number)]

    table_column_list = ['taxi_arrival', 'taxi_departure', 'online_arrival', 'online_departure', 'total_arrival',
                         'total_departure']

    point_dict_day_flow = {point_type: np.array(
        [point_dict_day_flow[point_type] for point_dict_day_flow in car_dict_point_dict_day_flow.values()]).sum(axis=0)
                           for
                           point_type in point_type_list}

    df_day_flow = pd.DataFrame({point_type + '_' + car_type: day_flow for car_type, point_dict_day_flow in
                                car_dict_point_dict_day_flow.items() for point_type, day_flow in
                                point_dict_day_flow.items()})

    week = range(week_first_day - 1, week_first_day + week_day_number - 1)

    week_string = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    df_week_day_flow = df_day_flow[(week_first_day - 1):(week_first_day + week_day_number - 1)]

    point_type_day_flow = pd.DataFrame(point_dict_day_flow)
    point_type_day_flow['week'] = (np.arange(month_day_number) + 3) % week_day_number
    average_week_day_flow_df = point_type_day_flow.groupby(by='week').mean()
    column_string_list = average_week_day_flow_df.columns.to_list()
    average_week_day_flow = average_week_day_flow_df.to_numpy().T

    point_dict_week_day_average_flow = {}
    for key, value in zip(column_string_list, average_week_day_flow):
        point_dict_week_day_average_flow[key] = value

    first_week_first_day = 5
    week_number = 3

    week_list = [range(first_week_first_day - 1 + week_day_number * i,
                       first_week_first_day + week_day_number - 1 + week_day_number * i) for i in range(3)]

    return {'month_day_number': month_day_number,
            'day_hour_number': day_hour_number,
            'car_dict_point_dict_day_hour_flow': car_dict_point_dict_day_hour_flow,
            'week_day_number': week_day_number,
            'day_list_point_dict_hour_flow': day_list_point_dict_hour_flow,
            'point_list_string': point_list_string,
            'day_list_dict_hour_flow': day_list_dict_hour_flow,
            'point_dict_day_flow': point_dict_day_flow,
            'df_day_flow': df_day_flow,
            'week': week,
            'df_week_day_flow': df_week_day_flow,
            'week_string': week_string,
            'average_week_day_flow_df': average_week_day_flow_df,
            'week_number': week_number,
            'week_list': week_list,
            'point_dict_week_day_average_flow': point_dict_week_day_average_flow,
            }


def pick_selected_df(df, flag_point_type):
    airport_min_lon = 116.573599
    airport_max_lon = 116.626814
    airport_min_lat = 40.047361
    airport_max_lat = 40.108833

    if flag_point_type == 'arrival':
        column = 'term'
    elif flag_point_type == 'departure':
        column = 'init'
    return df[(df[column + '_lon'] >= airport_min_lon) & (df[column + '_lon'] <= airport_max_lon) &
              (df[column + '_lat'] >= airport_min_lat) & (df[column + '_lat'] <= airport_max_lat)]


if __name__ == '__main__':
    # 1 需要绘制流量图
    # 2 需要绘制区域统计和聚类中心

    cluster_number = 10

    flag_point_type = 'arrival'
    taxi_df_list = read_df_list('taxi')
    taxi_df = pd.concat(taxi_df_list, axis=0, ignore_index=True)

    taxi_arrival_coordinate_df = cal_coordinate_df(taxi_df, flag_point_type=flag_point_type, is_output_target=True)
    taxi_arrival_count_df = taxi_arrival_coordinate_df.groupby(["lon", "lat"]).size().reset_index(name="time")
    taxi_arrival_count_ar = taxi_arrival_count_df.to_numpy()
    taxi_arrival_kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(taxi_arrival_coordinate_df)
    taxi_arrival_cluster_centers = taxi_arrival_kmeans.cluster_centers_

    taxi_arrival_12nd_day_coordinate_df = cal_coordinate_and_hour_df(taxi_df_list[11], flag_point_type=flag_point_type,
                                                                     is_output_target=True)
    hour_li_taxi_arrival_12nd_day_center_count = [(np.bincount(taxi_arrival_kmeans.predict(
        taxi_arrival_12nd_day_coordinate_df[taxi_arrival_12nd_day_coordinate_df['hour'] == i][['lon', 'lat']]),
        minlength=cluster_number)) for i in range(24)]
    hour_ar_center_flow = np.vstack(hour_li_taxi_arrival_12nd_day_center_count)

    airport_min_lon = 116.573599
    airport_max_lon = 116.626814
    airport_min_lat = 40.047361
    airport_max_lat = 40.108833

    airport_coord = np.array(
        [(airport_min_lon + airport_max_lon) / 2, (airport_min_lat + airport_max_lat) / 2]).reshape((1, 2))

    all_point_ar_coord = np.vstack((airport_coord, taxi_arrival_cluster_centers))
    import geopy.distance

    distance_2dar = [[geopy.distance.distance(coord_foo[::-1], coord_bar[::-1]).m for coord_foo in all_point_ar_coord]
                     for coord_bar in all_point_ar_coord]

    from lib.cal_bus_line import cal_line

    line_number = 4
    stop_ar_coord = taxi_arrival_cluster_centers

    stop_ar_gps_coord = taxi_arrival_cluster_centers
    hour_ar_bus_line = [cal_line(line_number=4, airport_coord_ar=airport_coord, stop_ar_coord_ar=stop_ar_coord,
                                     stop_ar_flow=center_flow, is_show_detail=False, is_show_iteration=False) for
                        center_flow in hour_ar_center_flow]

    # center_ar_certain_day_flow = hour_ar_center_flow.sum(axis=0)
    # certain_day_bus_line = cal_line(line_number=4, airport_coord_ar=airport_coord, stop_ar_coord_ar=stop_ar_coord,
    #                                 stop_ar_flow=center_ar_certain_day_flow, is_show_detail=True, is_show_iteration=False)



    stop_number = stop_ar_coord.shape[0]
    distance_array = np.linalg.norm(stop_ar_coord - airport_coord, ord=2, axis=1)
    # stop_sort_array = distance_array.argsort()
    # hour_ar_line_list_stop_index = [[stop_sort_array[bus_line == i] for i in range(line_number)] for bus_line in
    #                                 hour_ar_bus_line]
    hour_ar_line_list_stop_index = [[np.where(bus_line == i)[0] for i in range(line_number)]for bus_line in hour_ar_bus_line]
    hour_ar_line_list_stop_index = [[stop_index[distance_array[stop_index].argsort()] for stop_index in line_list_stop_index] for line_list_stop_index in hour_ar_line_list_stop_index]
    hour_ar_optimal_line_list_stop_coord = [[np.vstack((airport_coord[0, :], stop_ar_coord[v, :])) for v in
                                             line_list_stop_index] for line_list_stop_index in
                                            hour_ar_line_list_stop_index]

    result_line = {'cluster_centers': taxi_arrival_cluster_centers,
                   'hour_ar_center_flow': hour_ar_center_flow,
                   'airport_coord': airport_coord,
                   'line_number': line_number,
                   'hour_ar_optimal_line_list_stop_coord': hour_ar_optimal_line_list_stop_coord,
                   'hour_ar_center_flow': hour_ar_center_flow,
                   'distance_2dar': distance_2dar
                   }
    with open('out/result_line.pkl', 'wb') as f:
        pk.dump(result_line, f)

    pdb.set_trace()

    temp = time.time()

    df_list_dict = {}
    for flag_car_type in ['taxi', 'online']:
        print('----read_df_list----')
        print(f'flag_car_type:{flag_car_type}')
        df_list_dict[flag_car_type] = read_df_list(flag_car_type)

    temp2 = time.time()

    result_echart = cal_result_echart(df_list_dict)
    with open('out/result_echart.pkl', 'wb') as f:
        pk.dump(result_echart, f)

    temp3 = time.time()

    result_figure = cal_result_figure(df_list_dict)
    with open('out/result_figure.pkl', 'wb') as f:
        pk.dump(result_figure, f)

    print(temp2 - temp)
    print(temp3 - temp2)
    print(time.time() - temp3)
