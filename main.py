from worker import *

if __name__ == '__main__':
    def create_model():
        m, s = mx.new_model(), mx.new_space('s')

        @mx.defcells
        def b():
            pass

        @mx.defcells
        def a(n):
            import numpy as np
            return sum(sum(np.ones([n, n]) * np.ones([n, n]) * b()))

        m.save('m.mx')

    create_model()

    xw_mx_model_get('m.mx', 'm1')
    xw_mx_model_get('m.mx', 'm2')
    xw_mx_model_get('m.mx', 'm3')

    xw_mx_cell_set('m1','s','b', 1)
    xw_mx_cell_set('m2','s','b', 2)
    xw_mx_cell_set('m3','s','b', 3)

    from time import time, sleep

    sleep(2)# just to make sure that the workers are ready
    t0 = time()
    print('m1 = ', xw_mx_cell_get('m1','s','a', 10000))
    t1 = time()
    print('time 1 model:', t1 - t0)

    print('m2, m3 = ', xw_mx_cell_get(['m2', 'm3'],'s','a', 10000))
    print('time 2 models:', time() - t1)

    xw_mx_workers_stop()