from time import time
from numpy import partition, sum, argmin
from random import sample
from numpy.random import choice
from math import ceil
from multiprocessing import Pool, Queue
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


# TODO: minimize everything

class GreedyAlgorithm:
    def __init__(self, clusterization, measure = None):
        self. clusterization = clusterization
        if measure is None:
            self.measure = clusterization.init_measure()
        else:
            self.measure = measure

    def run(self):
        mutation_rate = 1
        while True:
            # candidates for the mutation
            centroids_numbers, centroid_distances = self.clusterization.get_nearest_centroids()
            to_mutate = n_mins(centroid_distances, mutation_rate)
            # mutation itself
            for point in to_mutate:
                new_measure = self.clusterization.recalculated_measure(point, centroids_numbers[point])
            if new_measure >= self.measure:
                break
            self.measure = new_measure
            mutation_rate *= 2
        return self.measure


class EvoOnePlusOne:
    def __init__(self, clusterization, measure = None):
        self. clusterization = clusterization
        if measure is None:
            self.measure = clusterization.init_measure()
        else:
            self.measure = measure

    def run(self):
        mutation_rate = 1
        start_time = time()
        while time() - start_time < 300:  # TODO think about the stopping criterion, now it is 5 minutes time
            # candidates for the mutation
            print("start iteration")
            centroids_numbers, centroid_distances = self.clusterization.get_nearest_centroids()

            # calculating the probabilities for the points to be moved
            sum_of_distances = sum(1/i for i in centroid_distances)
            probabilities = [(1 / i) / sum_of_distances for i in centroid_distances]

            # choosing each point with probability that is inversely proportional to its distance to the nearest cluster
            to_mutate = choice(list(range(len(centroids_numbers))), int(ceil(mutation_rate)), False, probabilities)

            new_measure = self.clusterization.recalculate_measure(to_mutate, [centroids_numbers[point] for point in to_mutate])

            if new_measure > self.measure: # Not sure, but it seems if it is equal we should divide the rate as well
                mutation_rate /= 2
            elif new_measure <= self.measure:
                mutation_rate *= 2 ** 0.25
                self.measure = new_measure
                self.clusterization.move_points(to_mutate, [centroids_numbers[point] for point in to_mutate])
            self.measure = new_measure
            print("Iteration " + str(self.measure))
        return self.measure


class EvoOnePlusFour:
    def __init__(self, clusterization, measure = None):
        self. clusterization = clusterization
        if measure is None:
            self.measure = clusterization.init_measure()
        else:
            self.measure = measure

    def mutation(self, mutation_rate):
        # this function must create a copy of the clusterization, generate a mutation, calculate the change in the
        # measure and return the new measure, the array of the moved points and the array of the clusters which
        # these points were moved to.
        clusterization = self.clusterization.copy()
        centroids_numbers, centroid_distances = clusterization.get_nearest_centroids()

        # calculating the probabilities for the points to be moved
        sum_of_distances = sum(1 / i for i in centroid_distances)
        probabilities = [(1 / i) / sum_of_distances for i in centroid_distances]

        # choosing each point with probability that is inversely proportional to its distance to the nearest cluster
        to_mutate = choice(list(range(len(centroids_numbers))), int(ceil(mutation_rate)), False, probabilities)
        return clusterization.recalculate_measure(to_mutate, [centroids_numbers[point] for point in to_mutate]), \
               to_mutate, [centroids_numbers[point] for point in to_mutate]

    def run(self):
        start_time = time()
        while time() - start_time < 300:  # TODO think about the stopping criterion, now it is 5 minutes time
            print("start iteration")

            with Pool(4) as pool:
                offspring = pool.map(self.mutation, [1, 4, 8, 16]) # creating four offspring in parallel threads

            best_offspring = argmin([child[0] for child in offspring])
            if offspring[best_offspring][0] <= self.measure:
                self.clusterization.move_points(*offspring[best_offspring][1:])  # actually moving the points, since
                                                                                 # we do not really do it in the
                                                                                 # mutation phase.
                self.measure = offspring[best_offspring[0]]

            print("Iteration " + str(self.measure))
        return self.measure

