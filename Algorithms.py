from time import time
from numpy import partition, sum
from random import sample
from numpy.random import choice
from math import ceil
#import Clusterization.clusterization as cl

# each algorithm takes some clusterization as an a input argument to the constructor
# it also can take the measure of this clusterization as an a
# each clusterization should have the following methods:
#
# measure() -- calcualte the measure of the clusterization (by the direct calculation).
#
# recalculated_measure(point_to_move, number_of_new_cluster) -- move point to another cluster and recalculate the
#                                                               measure
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
            self.measure = 0
        else:
            self.measure = measure

    def run(self):
        mutation_rate = 1
        while True:
            # candidates for the mutation
            centroids_numbers, centroid_distances = self.clusterization.get_nearest_centroids()
            to_mutate = n_mins([i for i in centroid_distances], mutation_rate)
            # mutation itself
            for point in to_mutate:
                new_measure = self.clusterization.recalculated_measure(point, centroids_numbers[point])
            if new_measure <= self.measure:
                break
            self.measure = new_measure
            mutation_rate *= 2
        return self.measure


class EvoOnePlusOne:
    def __init__(self, clusterization, measure = None):
        self. clusterization = clusterization
        if measure is None:
            self.measure = 0
        else:
            self.measure = measure

    def run(self):
        mutation_rate = 1
        start_time = time()
        while time() - start_time < 300:  # TODO think about the stopping criterion, now it is 5 minutes time
            # candidates for the mutation
            print("start iteration")
            centroids_numbers, centroid_distances = self.clusterization.get_nearest_centroids()
            sum_of_distances = sum(1/i for i in centroid_distances)
            probabilities = [(1 / i) / sum_of_distances for i in centroid_distances]
            # choosing each point with probability that is inversely proportional to its distance to the nearest cluster
            to_mutate = choice(list(range(len(centroids_numbers))), int(ceil(mutation_rate)), False, probabilities)
            # TODO Oh, fuck, we should be able to unmutate here... I'm done for now:(
            for point in to_mutate:
                new_measure = self.clusterization.recalculated_measure(point, centroids_numbers[point])
            if new_measure < self.measure: # Not sure, but it seems if it is equal we should divide the rate as well
                mutation_rate /= 2
            elif new_measure > self.measure:
                mutation_rate *= 2 ** 0.25
            self.measure = new_measure
            print("Iteration " + str(self.measure))
        return self.measure


