import os
import sys
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print(ROOT_DIR)
sys.path.append(os.path.dirname(ROOT_DIR))
from torchqtm.edbt.gens.sim_engine import DailySimulationClock
import unittest
import pandas as pd
import datetime

sessions = pd.DatetimeIndex(['20010101', '20010102'])
market_open = datetime.time(9, 30)
market_close = datetime.time(15, 00)
offet = 15

class Test_Clock(object):
    @staticmethod
    def test_0():
        it = DailySimulationClock(sessions, market_open, market_close, offet)
        print(list(it))
        

if __name__ == '__main__':
    Test_Clock.test_0()
    
    