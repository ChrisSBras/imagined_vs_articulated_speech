import json 
from statistics import stdev
if __name__ == "__main__":
    import sys

    filename = sys.argv[1]

    with open(filename, "r") as f:
        data = json.load(f)

        f.close()

        for item in data:
            accuracies = data[item]["accuracy"]
            avg = sum(accuracies) / len(accuracies)
            std = stdev(accuracies)

            print(item, f"${round(avg * 100, 1)} \\pm {round(100 * std, 1)}$")