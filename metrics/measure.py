from abc import abstractmethod, ABC


class Measure(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def find(self, X, labels, n_clusters):
        pass

    @abstractmethod
    def update(self, X, n_clusters, labels, k, l, id):
        pass
