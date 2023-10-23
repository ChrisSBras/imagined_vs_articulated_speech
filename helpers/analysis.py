import numpy as np
from helpers.io import mkdir_p

def count_correct_predictions(predicted: np.array, test_y: np.array, dir_name: str, filename: str) -> tuple:
    """
    Counts number of correct classifications given a list of predictions
    and a list of real answers.

    dir_name: directory to store results
    file_name: name for file to store results in

    return tuple(correct, total)
    """
    total = 0
    correct = 0
    wrong = 0

    mkdir_p(dir_name)
    with open(f"{dir_name}/{filename}.txt", "w") as f:
        for j, result in enumerate(predicted):
                y_hat = np.argmax(result) 
                y_real = np.argmax(test_y[j])

                f.write(f'{y_hat}, {y_real}\n')
                
                if (y_hat == y_real):
                    correct += 1
                else:
                    wrong += 1
                total += 1
    
    return correct, total