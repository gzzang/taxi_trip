# @Time    : 2020/4/19 9:48
# @Author  : gzzang
# @File    : cal_bus_line
# @Project : bus_route_planning


import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import pdb


def cal_bus_line(line_number, airport_coord, stop_ar_coord, stop_ar_flow, is_show_detail=True, is_show_iteration=True):
    def cal_line():
        def crossover(in_one, in_two):
            one = in_one.copy()
            two = in_two.copy()
            if np.random.rand() < crossover_probability:
                temp = np.random.rand(gene_number) < crossover_proportion
                one[temp] = in_two[temp]
                two[temp] = in_one[temp]
            return one, two

        def mutation(in_one):
            one = in_one.copy()
            if np.random.rand() < mutation_probability:
                temp = np.random.rand(gene_number) < mutation_proportion
                one[temp] = np.random.randint(low=0, high=line_number, size=[temp.sum()])
            return one

        def evaluate(in_one):
            return cal_distance(in_one)

        start_time = time.time()

        gene_number = stop_number
        generation_number = 64
        chromosome_number = 64
        crossover_probability = 0.4
        mutation_probability = 0.2
        crossover_proportion = 0.2
        mutation_proportion = 0.2

        pop = np.random.randint(low=0, high=line_number, size=[chromosome_number, gene_number])
        optimum_log = np.zeros(generation_number)

        for j in range(generation_number):
            if is_show_iteration:
                print(f'iteration:{j}')
            crossover_permutation = np.random.permutation(chromosome_number)
            crossover_pop = np.zeros(pop.shape)
            for i in range(int(chromosome_number / 2)):
                crossover_pop[2 * i], crossover_pop[2 * i + 1] = crossover(pop[crossover_permutation[2 * i]],
                                                                           pop[crossover_permutation[2 * i + 1]])
            mutation_pop = np.zeros(pop.shape)
            for i in range(chromosome_number):
                mutation_pop[i] = mutation(pop[i])

            new_pop = np.vstack((pop, crossover_pop, mutation_pop))

            value = np.array([evaluate(v) for v in new_pop])

            optimum_log[j] = value.min()
            optimal_chromosome_index = np.argmin(value)

            if j != generation_number - 1:
                minimal_value = value.min()
                maximal_value = value.max()

                temp1 = (maximal_value - value) / (maximal_value - minimal_value)
                temp2 = temp1 * temp1
                fitness = temp2.cumsum() / temp2.sum()

                target = np.random.rand(chromosome_number - 1)
                target.sort()
                final_index = np.zeros(chromosome_number - 1)

                target_i = 0
                fitness_i = 0
                for target_i in range(chromosome_number - 1):
                    while target[target_i] > fitness[fitness_i]:
                        fitness_i += 1
                    final_index[target_i] = fitness_i
                final_index = np.hstack((np.array([int(v) for v in final_index]), optimal_chromosome_index))
                pop = new_pop[final_index, :]
            else:
                optimal_chromosome = new_pop[optimal_chromosome_index]

        end_time = time.time()

        print(f'time:{end_time - start_time}')

        if is_show_detail:
            plt.figure()
            plt.plot(optimum_log)
            plt.xlabel('Iteration')

        return optimal_chromosome

    def cal_distance(xx):
        line_list_stop_index = [stop_sort_array[xx == i] for i in range(line_number)]
        line_list_stop_coord = [np.vstack((airport_coord[0, :], stop_ar_coord[v, :])) for v in line_list_stop_index]
        line_list_stop_flow = [stop_ar_flow[v] for v in line_list_stop_index]

        line_li_stop_sum_flow = [stop_flow.sum() for stop_flow in line_list_stop_flow]
        line_li_stop_distance_ar = [np.linalg.norm(v[1:] - v[:-1], axis=1) for v in line_list_stop_coord]
        line_li_cumsum_distance_ar = [np.cumsum(stop_distance)for stop_distance in line_li_stop_distance_ar]
        line_li_sum_distance = [np.sum(stop_distance) for stop_distance in line_li_stop_distance_ar]
        line_li_sum_cost = [np.sum(v*f) for v,f in zip(line_li_cumsum_distance_ar,line_list_stop_flow)]


        # actual_line_number = np.sum(np.array(line_li_sum_distance)>0)

        # print(np.sum(line_li_sum_cost))
        # print(stop_ar_flow.sum()/stop_number * np.sum(line_li_sum_distance)*2)
        # print('--')

        # np.sum(line_li_sum_distance)
        cost = np.sum(line_li_sum_cost) + stop_ar_flow.sum()/stop_number*2 * np.sum(line_li_sum_distance)
               #+ np.sum(np.array(line_li_sum_distance)*np.array(line_li_stop_sum_flow))

        # print(line_array_cost)

        # print(line_array_distance)
        # print(line_list_stop_flow)
        # print(line_list_stop_coord)



        # line_array_distance = np.array(
        #     [np.sum([vv * cumsum_flow for vv, cumsum_flow in zip(np.linalg.norm(v[1:] - v[:-1], axis=1), line_list_stop_flow)]) for v in
        #      line_list_stop_coord])

        # line_array_distance = np.array(
        #     [np.sum([vv * (v.shape[0] - vi) for vi, vv in enumerate(np.linalg.norm(v[1:] - v[:-1], axis=1))]) for v in
        #      line_list_stop_coord])

        # print(xx)
        # print(line_list_stop_coord)
        # pdb.set_trace()

        # line_array_distance = np.array(
        #     [np.sum(np.linalg.norm(v[1:] - v[:-1], axis=1)) for v in
        #      line_list_stop_coord])

        # distance = np.sum(line_array_distance)

        return cost

    stop_number = stop_ar_coord.shape[0]
    distance_array = np.linalg.norm(stop_ar_coord - airport_coord, ord=2, axis=1)
    stop_sort_array = distance_array.argsort()

    optimal_chromosome = cal_line()
    # if is_show_detail:

    return optimal_chromosome


if __name__ == '__main__':
    np.random.seed(2)
    max_flow = 100
    stop_number = 16
    airport_coord = np.random.rand(1, 2)
    stop_ar_coord = np.random.rand(stop_number, 2)
    stop_ar_flow = np.random.randint(low=0, high=max_flow, size=stop_number)

    np.random.seed(int(time.time()))
    # with open('out/data.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #
    #
    # airport_coord = data['airport_coord']
    # stop_ar_coord = data['stop_ar_coord']
    # stop_ar_flow = data['stop_ar_flow']

    line_number = 8
    bus_line = cal_bus_line(line_number, airport_coord, stop_ar_coord, stop_ar_flow)

    stop_number = stop_ar_coord.shape[0]
    distance_array = np.linalg.norm(stop_ar_coord - airport_coord, ord=2, axis=1)
    stop_sort_array = distance_array.argsort()
    line_list_stop_index = [stop_sort_array[bus_line == i] for i in range(line_number)]
    optimal_line_list_stop_coord = [np.vstack((airport_coord[0, :], stop_ar_coord[v, :])) for v in
                                    line_list_stop_index]

    plt.figure()
    plt.scatter(airport_coord[0, 0], airport_coord[0, 1], c='r', s=20, alpha=0.5)
    for i in range(line_number):
        plt.plot(optimal_line_list_stop_coord[i][:, 0], optimal_line_list_stop_coord[i][:, 1])
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])

    plt.show()
