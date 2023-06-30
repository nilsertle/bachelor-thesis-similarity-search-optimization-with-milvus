import numpy as np

def calc_recall_rate(gt: np.ndarray, results: np.ndarray):
    '''
    The performance measure is recall@R, that is, for varying values of R, the average rate of queries for which the 1-nearest neighbor is ranked in the top R positions. Please use this measure to allow a direct comparison of your system with most of the results reported in the literature. 
    '''
    recall_rates = []
    for r in [1,10,100]:
        recall_rate = 0
        for i in range(len(results)):
            for j in range(r):
                if results[i][j] == gt[i][0]:
                    recall_rate += 1
                    break
        recall_rate /= len(results)
        recall_rates.append(recall_rate)
    return recall_rates


