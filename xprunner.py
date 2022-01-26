import os
import gc
import sys
import time
import datetime
import pytz
import uuid
import shutil
import random
import logging
import IPython
from dataclasses import dataclass
from abc import abstractmethod

__keras_tuner__ = False
__project_name__ = 'xptracker'
__data_dir__ = 'xpdata'

__version__ = 0.004


def prepare_dev_stationary():
    global DEV
    """ 
    DEV_DATA - location of shared development folder on google drive 
    """
    global DEV_DATA
    try:
        test_ipython = str(get_ipython())
    except NameError:
        print('Running on local environment')
        DEV = os.getcwd()
    else:
        if 'google.colab' in test_ipython:
            print('Running on CoLab')
            from google.colab import drive
            drive.mount('/content/drive')
            if __keras_tuner__:
                cmd = "pip install -U keras-tuner"
                os.system(cmd)
            DEV = os.path.join('/content/drive/MyDrive/Python/', __project_name__)
        elif 'ipykernel' in test_ipython:
            print('Running on Jupyter Notebook')
            DEV = os.getcwd()
    sys.path.append(DEV)
    path_head = os.path.split(DEV)[0]
    DEV_DATA = os.path.join(path_head, __data_dir__)
    pass


prepare_dev_stationary()


random.seed(42)

logger = logging.getLogger(__name__)


def get_local_timezone_name():
    if time.daylight:
        offset_hour = time.altzone / 3600
    else:
        offset_hour = time.timezone / 3600

    offset_hour_msg = f"{offset_hour:.0f}"
    if offset_hour > 0:
        offset_hour_msg = f"+{offset_hour:.0f}"
    return f'Etc/GMT{offset_hour_msg}'


TIMEZONE = pytz.timezone(get_local_timezone_name())


@dataclass()
class XPConfig:
    root_dir = DEV_DATA
    queue_dir = os.path.join(root_dir, "queue")
    done_dir = os.path.join(root_dir, "done")
    work_dir = os.path.join(root_dir, "work")
    semaphore_dir = os.path.join(root_dir, 'temp')
    xperiments = "xperiments"


class SignalMan(object):
    signal_id = uuid.uuid1(uuid.getnode())

    def __init__(self,
                 sleep_time_range=(25, 90)):
        self.id = self.__class__.signal_id
        self.wait_flag_fname = f'{self.id.hex}.w8'
        self.path_wait_flag_fname = os.path.join(XPConfig.semaphore_dir, f'{self.id.hex}.w8')
        self.sleep_time_range = sleep_time_range
        self.sleep_time: int = self.get_sleep_time()
        print(self.id)
        self.__check_crashed_flags()

    def __check_crashed_flags(self):
        wait_list = self.get_flags()
        if len(wait_list) == 1 and wait_list[0] == self.wait_flag_fname:
            """ Checking if this one our OLD wait flag """
            msg = f'Founded this instance OLD wait flag. Previous instance is crashed?!'
            logger.debug(msg)
            self.remove_wait_flag()
        else:
            """ Checking if any crashed flags exists for multiuser version 
                write code here!!! 
            """
            pass
        pass

    def remove_wait_flag(self):
        try:
            os.remove(self.path_wait_flag_fname)
        except FileNotFoundError as e:
            msg = f'Error : {self.path_wait_flag_fname} not_found. Something wrong?!'
            logger.debug(msg)
        pass

    def wait(self):
        print(self.sleep_time)
        time.sleep(self.sleep_time)
        self.sleep_time = self.get_sleep_time()

    def get_flags(self):
        wait_list = []
        for fname in os.listdir(XPConfig.semaphore_dir):
            if fname.endswith('.w8'):
                wait_list.append(fname)
        return wait_list

    def is_wait_flag(self):
        wait_list = self.get_flags()
        if wait_list:
            return True
        else:
            return False

    def set_wait_flag(self):
        if not self.is_wait_flag():
            open(self.path_wait_flag_fname, 'a').close()
            time.sleep(self.sleep_time)
        """ setting the flag and waiting if the flag is not alone """
        wait_list = self.get_flags()
        if len(wait_list) > 1:
            """ Remove our flag to solve flags conflict with different time intervals """
            self.remove_wait_flag()
            return False
        return True

    def get_sleep_time(self):
        return random.randint(self.sleep_time_range[0], self.sleep_time_range[1])


class EnvVM:
    """ Environment to create object to run"""

    def __init__(self,
                 run_object_name='TrainNN'):
        self.run_object_name = run_object_name
        pass

    def __setup(self):
        pass

    def run(self, job_dir):
        pass


class JobRunner:
    def __init__(self):
        self.job_list = []
        self.job_dir: str = ''
        self.env = EnvVM()
        pass

    def is_new_job_exist(self):
        """
        at 1st versions we check for new jobs queue directory only
        """
        logger.info(f'Checking if new job exist...')
        dir_list = os.listdir(XPConfig.queue_dir)
        job_list = []
        for fname in dir_list:
            if os.path.isdir(fname):
                job_list.append(fname)
        if job_list:
            self.job_list = job_list
            return True
        else:
            return False

    def __copy_job_to_work_dir(self):
        shutil.copytree(os.path.join(XPConfig.queue_dir, self.job_dir), XPConfig.work_dir)
        pass

    def get_job(self):
        if len(self.job_list) > 1:
            self.job_dir = random.sample(self.job_list, 1)
        else:
            self.job_dir = self.job_list[0]
        logger.info(f'Getting new job #{self.job_dir}')
        pass

    def setup_job(self):
        pass

    def run_job(self):
        logger.info(f'Starting job #{self.job_dir}')
        pass

    def finish_job(self):
        logger.info(f'Finishing job #{self.job_dir}')
        self.job_dir = ''
        self.job_list = []
        self._reset()
        pass

    def _reset(self):
        gc.collect()
        pass


class Runner:
    def __init__(self,
                 sleep_time_range=(25, 90),
                 verbose=1):

        self.root_dir = XPConfig.root_dir
        self.done_dir = XPConfig.done_dir
        self.queue_dir = XPConfig.queue_dir
        self.in_progress_dir = XPConfig.work_dir
        self.xperiments = XPConfig.xperiments

        self.__setup()
        self.semaphore = SignalMan(sleep_time_range=sleep_time_range)
        self.log_name = f'{self.semaphore.id.hex}.log'
        self._set_logging(verbose)
        logger.info(f'Started...')
        self.worker = JobRunner()
        self.run()
        pass

    def __check_create(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        pass

    @abstractmethod
    def __setup(self):
        """
        Checking directory setup and create directories if needed
        """
        self.__check_create(XPConfig.root_dir)
        self.__check_create(XPConfig.queue_dir)
        self.__check_create(XPConfig.done_dir)
        self.__check_create(XPConfig.work_dir)
        self.__check_create(XPConfig.semaphore_dir)


        pass

    def _set_logging(self, verbose):
        if verbose == 0:
            logger.setLevel(logging.NOTSET)
        elif verbose == 1:
            logger.setLevel(logging.DEBUG)
        elif verbose == 2:
            logger.setLevel(logging.INFO)
        log_format = '%(asctime)s - %(message)s'
        path_filename = os.path.join(XPConfig.root_dir, self.log_name)
        file_handler = logging.FileHandler(path_filename)
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def run(self):
        while True:

            if self.semaphore.is_wait_flag():
                self.semaphore.wait()
            else:
                print('check1')
                if self.semaphore.set_wait_flag():
                    if self.worker.is_new_job_exist():
                        self.worker.get_job()
                        self.worker.setup_job()
                        self.semaphore.remove_wait_flag()
                        self.worker.run_job()
                        self.worker.finish_job()
                    else:
                        self.semaphore.remove_wait_flag()
                        self.semaphore.wait()
                else:
                    self.semaphore.wait()


if __name__ == '__main__':
    rn = Runner(sleep_time_range=(5, 30),
                verbose=2)
