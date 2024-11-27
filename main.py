from map import Map
import timeit
import numpy as np

test: Map = Map(100, 100)

data: np.ndarray = np.array([2, 4, 5, 2, 4, 6, 7, 2, 3, 4])

test.add_character(1, 2, 2, data)
test.move_character(1, 0, 1)
print(test.get_surroundings(1))