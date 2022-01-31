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
from objrun import ObjectRunner
import importlib.util
import importlib.machinery
from distutils.dir_util import copy_tree
import optuna


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
        DEV_DATA = os.path.join(DEV, __data_dir__)
    else:
        if 'google.colab' in test_ipython:
            print('Running on CoLab')
            from google.colab import drive
            drive.mount('/content/drive')
            if __keras_tuner__:
                cmd = "pip install -U keras-tuner"
                os.system(cmd)
            DEV = os.path.join('/content/drive/MyDrive/Python/', __project_name__)
            path_head = os.path.split(DEV)[0]
            DEV_DATA = os.path.join(path_head, __data_dir__)
        elif 'ipykernel' in test_ipython:
            print('Running on Jupyter Notebook')
            DEV = os.getcwd()
            DEV_DATA = os.path.join(DEV, __data_dir__)
    sys.path.append(DEV)
    pass


def get_local_timezone_name():
    if time.daylight:
        offset_hour = time.altzone / 3600
    else:
        offset_hour = time.timezone / 3600

    offset_hour_msg = f"{offset_hour:.0f}"
    if offset_hour > 0:
        offset_hour_msg = f"+{offset_hour:.0f}"
    return f'Etc/GMT{offset_hour_msg}'


prepare_dev_stationary()

TIMEZONE = pytz.timezone(get_local_timezone_name())

random.seed(42)

logger = logging.getLogger(__name__)


class EnvTrack:
    """ Environment to create object to run"""
    def __init__(self,
                 work_dir: str = ".",
                 import_module_name: str = 'xprun',
                 run_object_name: str = 'TrainNN'
                 ):
        self.import_module_name = import_module_name
        self.run_object_name: str = run_object_name
        self.work_dir = work_dir
        self.module = None
        self.ClassObj = None
        self.objInstance = None
        pass

    def __import_job(self):
        self.loader = importlib.machinery.SourceFileLoader(self.import_module_name, os.path.join(self.work_dir, f'{self.import_module_name}.py'))
        self.spec = importlib.util.spec_from_loader(self.import_module_name, self.loader)
        self.module = importlib.util.module_from_spec(self.spec)
        self.loader.exec_module(self.module)
        setattr(self, 'ClassObj', getattr(self.module, self.run_object_name))
        print()

    def setup(self):
        self.__import_job()
        self.objInstance = self.ClassObj()
        pass

    def run(self):
        self.objInstance.run()
        pass

    def reset(self):
        pass


if __name__ == '__main__':
    env = EnvTrack(work_dir="examples/condenrock_ny2022",
                   import_module_name='train',
                   run_object_name='ObjTest')
    env.setup()
    env.run()
