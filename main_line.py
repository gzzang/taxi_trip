# @Time    : 2020/4/19 10:21
# @Author  : gzzang
# @File    : main_line
# @Project : taxi_trip

import pickle as pk
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb

path = 'out/'
out_path='out/line/'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

with open(path+'result_line.pkl', 'rb') as f:
    result_line = pk.load(f)

cluster_centers = result_line['cluster_centers']
hour_ar_center_flow = result_line['hour_ar_center_flow']
airport_coord = result_line['airport_coord'].reshape((1, 2))
hour_ar_optimal_line_list_stop_coord = result_line['hour_ar_optimal_line_list_stop_coord']
line_number = result_line['line_number']
hour_ar_center_flow = result_line['hour_ar_center_flow']
distance_2dar = result_line['distance_2dar']

index = 0
for optimal_line_list_stop_coord,center_flow in zip(hour_ar_optimal_line_list_stop_coord, hour_ar_center_flow):
    plt.figure()
    for i in range(line_number):
        plt.plot(optimal_line_list_stop_coord[i][:, 0], optimal_line_list_stop_coord[i][:, 1])
    plt.scatter(airport_coord[0, 0], airport_coord[0, 1], c='black', s=50, alpha=1, marker='*')
    plt.scatter(cluster_centers[:,0],cluster_centers[:,1], c='black', s=center_flow, alpha=1)

    # for i,coord in enumerate(cluster_centers):
    #     plt.annotate(f'{i+1}',xy=coord, xytext=(0, 8), textcoords='offset points', ha='center', va='bottom')

    plt.xlabel(u'经度')
    plt.ylabel(u'纬度')
    plt.savefig(out_path+f'figure_{index}.png')
    plt.close()
    index = index + 1


pd.DataFrame(hour_ar_center_flow).to_csv(out_path + 'flow.csv')
pd.DataFrame(cluster_centers).to_csv(out_path + 'center_coord.csv')
pd.DataFrame(np.around(np.array(distance_2dar)/1000, decimals=2)).to_csv(out_path + 'distance.csv')

