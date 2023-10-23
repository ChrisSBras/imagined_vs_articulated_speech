"""
Confusion matrix generator script
"""

import os
import sys


def matrix_print(matrix):
    for l in matrix:
        print(l)

if __name__ == "__main__":
    directory = sys.argv[1]
    files = os.listdir(directory)

    num_targets = 6
    for name in files:
        matrix = [[0 for j in range(num_targets)] for i in range(num_targets) ]

        with open(f"./{directory}/{name}", "r") as f:
            lines = f.readlines()
            total = 0
            correct = 0
            for line in lines:
                split = line.split(",")
                pred = int(split[0])
                true = int(split[1])

                total += 1
                if pred == true:
                    correct += 1
                matrix[true][pred] += 1
        print(name)
        print(f"ACCURACY {(correct/total) * 100:.2f}%")
        matrix_print(matrix)

        print()

