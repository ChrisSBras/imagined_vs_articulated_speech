import numpy as np

def count_correct_predictions(predicted: np.array, real: np.array) -> tuple:
    """
    Counts number of correct classifications given a list of predictions
    and a list of real answers.

    return tuple(correct, total)
    """
    total = 0
    correct = 0
    wrong = 0

    for i, result in enumerate(predicted):
        y_hat = np.argmax(result) 
        y_real = np.argmax(real[i])
        
        if (y_hat == y_real):
            correct += 1
        else:
            wrong += 1
        total += 1
    
    return correct, total