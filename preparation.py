# @Time    : 2020/3/30 18:01
# @Author  : gzzang
# @File    : preparation
# @Project : taxi_trip

import numpy as np
import pandas as pd

import pdb



def drop_noisy(df, columns_str):
    df_describe = df.describe()
    for column in columns_str:
        mean = df_describe.loc['mean', column]
        std = df_describe.loc['std', column]
        minvalue = mean - 4 * std
        maxvalue = mean + 4 * std
    return df[(df[column] >= minvalue) & (df[column] <= maxvalue)]

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

def get_standard_df(file_path):
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
    return standard_df

def cal_coordinate_count_of_one_day(file_path, is_airport_initiation=True, is_output_airport=False):
    coordinate_list = ['init', 'term']
    airport_coordinate_type = coordinate_list[not is_airport_initiation]
    another_coordinate_type = coordinate_list[is_airport_initiation]

    standard_df = get_standard_df(file_path)

    airport_df = get_airport(standard_df, airport_coordinate_type)

    if is_output_airport:

        count_point_df = airport_df.groupby(
            [airport_coordinate_type + "_longitude", airport_coordinate_type + "_latitude"]).size().reset_index(
            name="time")
    else:
        count_point_df = airport_df.groupby(
            [another_coordinate_type + "_longitude", another_coordinate_type + "_latitude"]).size().reset_index(
            name="time")
    count_point_df.columns = ['longitude', 'latitude', 'time']

    return count_point_df


def cal_coordinate_of_one_day(file_path, is_airport_initiation=True, is_output_airport=False):
    coordinate_list = ['init', 'term']
    airport_coordinate_type = coordinate_list[not is_airport_initiation]
    another_coordinate_type = coordinate_list[is_airport_initiation]

    standard_df = get_standard_df(file_path)

    airport_df = get_airport(standard_df, airport_coordinate_type)

    if is_output_airport:
        coordinate = airport_df[[airport_coordinate_type + "_longitude", airport_coordinate_type + "_latitude"]].to_numpy()
    else:
        coordinate = airport_df[[another_coordinate_type + "_longitude", another_coordinate_type + "_latitude"]].to_numpy()

    return coordinate


if __name__ == '__main__':
    file_path = 'data/taxi/20180301.csv'
    print(cal_coordinate_of_one_day(file_path, is_airport_initiation=True, is_output_airport=False))