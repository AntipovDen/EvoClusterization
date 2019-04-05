import gc
from time import time
from numpy import partition, sum, argmin
from random import sample
from numpy.random import choice
from math import ceil
from multiprocessing import Pool, Queue
from sys import float_info, stderr
#import Clusterization.clusterization as cl

# each algorithm takes some clusterization as an a input argument to the constructor
# it also can take the measure of this clusterization as an a
# each clusterization should have the following methods:
#
# measure() -- calcualte the measure of the clusterization (by the direct calculation).
#
##### recalculated_measure(point_to_move, number_of_new_cluster) -- move point to another cluster and recalculate the
#####                                                               measure
# NEW recalculate_measure(pints_to_move, clusters_to_move_to) -- calculate the measure that will be after moving the
#                                                                set of points to the specified clusters, but the
#                                                                labels of the points stay the same!
#
# NEW move_points(pints_to_move, clusters_to_move_to) -- move points to the neighbour clusters (without any calculation
#                                                        of the measure.
#
# get_nearest_centroids() -- returns two arrays. The first array contains the numbers of the cluster with the nearest
#                            centroid for each point. The second array contains the distance to this cluster.
#
# It will be great to store the current measure in the clusterization, if it is already calculated, maybe...


def n_mins(arr, n):  # returns the indices of the n smallest elements in the argument, breaking ties at random.
    nth_smallest = partition(arr, n)[n]
    res_1 = [i for i in range(len(arr)) if arr[i] < nth_smallest]
    res_2 = [i for i in range(len(arr)) if arr[i] == nth_smallest]
    return res_1 + sample(res_2, n - len(res_1))


# all measures are supposed to be minimized

class GreedyAlgorithm:
    def __init__(self, clusterization, measure = None):
        self. clusterization = clusterization
        if measure is None:
            self.measure = clusterization.init_measure()
        else:
            self.measure = measure

    def run(self):
        start_time = time()
        mutation_rate = 1
        iter = 0
        while True:
            # candidates for the mutation
            print("iteration\t{}".format(iter))
            print("measure\t\t{}".format(self.measure))
            centroids_numbers, centroid_distances = self.clusterization.get_nearest_centroids()
            to_mutate = n_mins(centroid_distances, mutation_rate)
            print("mutation\t{}".format(to_mutate))
            # mutation itself
            new_measure = self.clusterization.recalculated_measure_C(to_mutate, centroids_numbers)
            if new_measure >= self.measure:
                break
            self.clusterization.move_points()
            self.measure = new_measure
            mutation_rate *= 2
            iter += 1
        return self.measure, iter, time() - start_time


class EvoOnePlusOne:
    def __init__(self, clusterization, measure = None):
        self. clusterization = clusterization
        if measure is None:
            self.measure = clusterization.init_measure()
        else:
            self.measure = measure

    def run(self):
        start_time = time()
        mutation_rate = 1
        iter = 0
        while time() - start_time < 300:  # TODO think about the stopping criterion, now it is 5 minutes time
            # candidates for the mutation
            print("iteration\t{}".format(iter))
            print("measure\t\t{}".format(self.measure))
            centroids_numbers, centroid_distances = self.clusterization.get_nearest_centroids()

            # calculating the probabilities for the points to be moved
            sum_of_distances = sum(1/i for i in centroid_distances)
            probabilities = [(1 / i) / sum_of_distances for i in centroid_distances]

            # choosing each point with probability that is inversely proportional to its distance to the nearest cluster
            to_mutate = choice(list(range(len(centroids_numbers))), int(ceil(mutation_rate)), False, probabilities)
            print("mutation\t{}".format(to_mutate))

            new_measure = self.clusterization.recalculated_measure_C(to_mutate, [centroids_numbers[point] for point in to_mutate])
            print("new measure\t{}".format(new_measure))

            if new_measure > self.measure:
                mutation_rate = min(mutation_rate * 2 ** 0.25, len(self.clusterization.labels) / 2)
                print("declined")
            elif new_measure <= self.measure:
                mutation_rate = max(mutation_rate / 2, 1)
                print("accepted")
                self.measure = new_measure
                self.clusterization.move_points()

            print("new rate\t{}".format(mutation_rate))
            iter += 1
            # print("Iteration " + str(self.measure))
        return self.measure, iter, time() - start_time


class EvoOnePlusFour:
    def __init__(self, clusterization, measure = None):
        self. clusterization = clusterization
        if measure is None:
            self.measure = clusterization.init_measure()
        else:
            self.measure = measure

    def mutation(self, mutation_rate):
        # this function must generate a mutation, calculate the change in the
        # measure and return the new measure, the array of the moved points and the array of the clusters which
        # these points were moved to.
        try:
            centroids_numbers, centroid_distances = self.clusterization.get_nearest_centroids()

            # calculating the probabilities for the points to be moved
            sum_of_distances = sum(1 / i for i in centroid_distances)
            probabilities = [(1 / i) / sum_of_distances for i in centroid_distances]

            # choosing each point with probability that is inversely proportional to its distance to the nearest cluster
            to_mutate = choice(list(range(len(centroids_numbers))), int(ceil(mutation_rate)), False, probabilities)
            return self.clusterization.recalculated_measure_parallel(to_mutate, [centroids_numbers[point] for point in to_mutate])
        except MemoryError:
            print("Thread with mutation rate {} has tragically died".format(mutation_rate), file=stderr)
            return float_info.max, None, None
        # Notice: clusterization.recalculated_measure_parallel returns not only new measure, but the copy of the
        # labels and of the measure.

    def run(self):
        start_time = time()
        iter = 0
        while time() - start_time < 300:  # TODO think about the stopping criterion, now it is 5 minutes time
            print("iteration\t{}".format(iter))
            print("measure\t\t{}".format(self.measure))
            print("running garbage collector")
            offspring = None
            gc.collect()
            print("garbage collector must have done its work")

            with Pool(4) as pool:
                offspring = pool.map(self.mutation, [2 ** i for i in range(4)]) # creating four offspring in parallel threads

            print("mut rate\toffsring measure")
            for i in range(4):
                print("{}\t\t\t{}".format(2 ** i, offspring[i][0]))

            best_offspring = argmin([child[0] for child in offspring])
            if offspring[best_offspring][0] <= self.measure:
                self.clusterization.move_points(*offspring[best_offspring][1:])  # actually moving the points, since
                                                                                 # we do not really do it in the
                                                                                 # mutation phase.
                self.measure = offspring[best_offspring][0]
                print("accepted")
                print("new measure\t{}".format(self.measure))
            else:
                print("declined")
            # print("Iteration " + str(self.measure))
            iter += 1
        return self.measure, iter, time() - start_time

