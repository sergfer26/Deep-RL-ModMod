from SALib.sample import saltelli
import numpy as np
import pandas as pd


def controls_to_csv(problem, n):
    controles = saltelli.sample(problem, n,calc_second_order=True)
    data = pd.DataFrame(controles, columns=("U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8", "U9", "U10"))
    data.to_csv("CONTROLES.csv", index=False)
    
def main():
    '''
    saltelli_sequence = np.zeros([(2 * Dg + 2) * N, D])
    '''
    problem = {"num_vars": 10,"names": ["U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8", "U9", "U10"],
               "bounds":  [[x,y] for x,y in zip(0*np.ones(10),np.ones(10))]}
    N = 512 #A performance comparison of sensitivity analysis methods for building energy models
    controls_to_csv(problem, N)

if __name__ == "__main__":
    main()