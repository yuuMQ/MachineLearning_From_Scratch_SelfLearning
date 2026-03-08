import numpy as np
import math
import sys

# Euclid and Manhattan distance
def euclid_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2

    return math.sqrt(distance)

def manhattan_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += abs(x1[i] - x2[i])

    return distance