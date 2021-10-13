import numpy as np
import time

from pdag.data_generator import generate_random_partialB, generate_random_MVN_data
from pdag.pdag import PDAG

if __name__ == '__main__':
    p = 20
    n = int(p/2)
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
    Bhat = pdag4.fit(X, 1.0, int(p/4), int(2*p/4), int(3*p/4))

    for p in [100,500]:
        print("For p = ", p, end=' ')
        n = int(p/2)
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

        start = time.time()
        Bhat = pdag4.fit(X, 0.2, int(p/4), int(2*p/4), int(3*p/4))
        end = time.time()
        print(",elapsed  (after compilation) = %s" % (end - start), end=' ')

        print('and number of non-zero off-diagonal elements are: ', np.sum(np.round(Bhat,6)!=0)-p)