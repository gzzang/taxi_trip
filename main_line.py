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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

with open('out/result_line.pkl', 'rb') as f:
    result_line = pk.load(f)

cluster_centers = result_line['cluster_centers']
hour_ar_center_flow = result_line['hour_ar_center_flow']
airport_coord = result_line['airport_coord'].reshape((1, 2))
optimal_line_list_stop_coord = result_line['optimal_line_list_stop_coord']
line_number = result_line['line_number']

plt.figure()
for i in range(line_number):
    plt.plot(optimal_line_list_stop_coord[i][:, 0], optimal_line_list_stop_coord[i][:, 1], marker='o', markersize=2)
plt.scatter(airport_coord[0, 0], airport_coord[0, 1], c='black', s=50, alpha=1, marker='*')
plt.xlabel(u'经度')
plt.ylabel(u'纬度')
plt.savefig('out/line/figure.png')
plt.show()
