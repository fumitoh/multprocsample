# Based on sample code contributed by Alexey Baran

from worker import *

if __name__ == '__main__':

    no_proc = 10

    for i in range(1, no_proc+1):
        xw_mx_model_get(r'C:\Path\to\model10x', 'm' + str(i))

    from time import time, sleep

    sleep(2)# just to make sure that the workers are ready
    t0 = time()

    print('result = ', xw_mx_cell_get(['m' + str(i) for i in range(1, no_proc+1)], 'Projection', 'PV_TotalNetCashflow', 0))
    print('time models:', time() - t0)

    xw_mx_workers_stop()