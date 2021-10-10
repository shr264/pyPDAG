# CSCS

This implements the Partition-DAG algorithm for covariance and DAG estimation from [ESTIMATION OF GAUSSIAN DIRECTED ACYCLIC GRAPHS USING PARTIAL ORDERING INFORMATION WITH APPLICATIONS TO DREAM3 NETWORKS AND DAIRY CATTLE DATA](https://arxiv.org/pdf/1902.05173.pdf) by Kshitij Khare, Sang Oh, Syed Rahman and Bala Rajaratnam

## Basic scripts

The program consists of the following scripts
* data_generate.py: used to generate random multivariate data accoriding to a graph
* PDAG.py: contains the main functions and class for Partition-DAG to estimate DAG
* main.py: runs the program to generate the DAG

## Notebooks

A notebook with an example is also included
* PDAG_example.ipynb

## Example

```
import numpy as np
import time
from pdag.data_generator import generate_random_partialB, generate_random_MVN_data
from pdag.pdag import PDAG

np.random.seed(3689)
p = 200
n = 10*p
omega, B, A, G = generate_random_partialB(int(p/4), 
                                            int(2*p/4), 
                                            int(3*p/4),
                                            p = p,
                                            a = 0.3,
                                            b = 0.7,
                                            diag_a = 2,
                                            diag_b = 5,
                                            plot=False)
B = np.array(B)
X = generate_random_MVN_data(n, omega)

pdag4 = PDAG(4)
start = time.time()
Bhat = pdag4.fit(X, 0.1, int(p/4), int(2*p/4), int(3*p/4))
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))
```

## Authors

* **Syed Rahman**