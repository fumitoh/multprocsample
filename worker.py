from multiprocessing import Process, Pipe, Queue
import modelx as mx
from functools import wraps
import logging
import logging.config
import inspect

p_to_reader_dict = {}
p_from_reader_dict = {}
workers = {}

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
log = logging.getLogger(__name__)





def xw_mx_model_get(model_path, model_id):
    # Pipes are unidirectional with two endpoints:  p_input ------> p_output
    if model_id not in p_to_reader_dict:
        q = Queue()
        p_from_reader_output, p_from_reader_input = Pipe()
        worker = ModelWorker(q, p_from_reader_input, model_path, model_id)
        worker.daemon = True
        worker.start()  # Launch the reader process
        p_to_reader_dict[model_id] = q
        p_from_reader_dict[model_id] = p_from_reader_output
        workers[model_id] = worker
    else:
        log.warning("Model %s already started and still open..." % model_id)
    return True


def xw_mx_workers_stop():
    for model_id in workers:
        if model_id in p_to_reader_dict:
            p_to_reader_dict[model_id].put(("STOP",))
            del p_to_reader_dict[model_id]



def mx_worker(response=True):
    def callable(f):
        @wraps(f)
        def wrapper(models, *args):
            responses = []
            models_list = (models if isinstance(models, list) else [models])
            for model_id in models_list:
                if model_id in p_to_reader_dict:
                    p_to_reader_dict[model_id].put((f(models, *args), *args))

            for model_id in models_list:
                if model_id not in p_to_reader_dict:
                    r = False
                else:
                    r = p_from_reader_dict[model_id].recv() if response else True
                responses.append(r)
            return responses if len(responses) > 1 else responses[0]
        return wrapper
    return callable


class ModelWorker(Process):
    def __init__(self, task, result, model_path, model_id):
        Process.__init__(self)
        self.task = task
        self.result = result
        self.model = None
        self.space = None
        self.model_path = model_path
        self.model_id = model_id

    def stop_worker(self):
        log.warning("Terminating worker for %s", self.model_id)
        self.terminate()

    def open_model(self):
        self.model = mx.restore_model(self.model_path, name=self.model_id)

    def get_space(self, space_names):
        spaces = []
        for space_name in (space_names if isinstance(space_names, list) else [space_names]):
            spaces.append(eval("mx.models['%s'].%s" % (self.model_id, space_name)))
        return spaces if len(spaces) > 1 else spaces[0]


    def run(self):
        log.warning("Starting worker for %s ..." % self.model_id)
        self.open_model()
        task_id = 0
        while True:
            args = self.task.get()  # Read from the output pipe and do nothing
            task_id += 1
            action = args[0]
            if callable(action):
                import time
                start = time.time()
                log.debug("%s task %i: %s START, args: \n %s" % (self.model_id, task_id, action.__name__, (*args[1:],)))
                try:
                    args_id = 1
                    first_arg = inspect.getfullargspec(action)[0][0]
                    if first_arg[:5] == 'space':
                        mx_obj = self.get_space(args[1])
                        args_id += 1
                    elif first_arg == 'model':
                        mx_obj = self.model
                    else:
                        raise NotImplementedError('Unknown first arg')

                    response = action(mx_obj, *args[args_id:])
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("Unexpected error :", str(e))
                    response = False
                end = time.time()
                log.debug("%s task %i: %s FINISH" % (self.model_id, task_id, action.__name__))
                log.debug("%s task %i: CALCULATION TIME: %s" % (self.model_id, task_id, str(end-start)))
                if response is not None:
                    self.result.send(response)
            elif action == 'STOP':
                break
            else:
                raise ValueError('wrong action: ', action)

        log.warning('Terminating worker for %s' % self.model_id)


def cell_get(space, cell_name, *cell_args):
    return space.cells[cell_name](*cell_args)

@mx_worker(response=True)
def xw_mx_cell_get(model_id, space, cell_name, *cell_args):
    return cell_get


def cell_set(space, cell_name, value, *cell_args):
    space.cells[cell_name][cell_args] = value

@mx_worker(response=False)
def xw_mx_cell_set(model_id, space, cell_name, value, *cell_args):
    return cell_set