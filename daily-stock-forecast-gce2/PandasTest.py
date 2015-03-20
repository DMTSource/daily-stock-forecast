#Test pandas for downloading historicla into a dataframe

import pandas as pd
import pandas.io.data
from pandas import Series, DataFrame

from datetime import datetime, time

from GetSymbols import *

if __name__ == "__main__":

    fullSymbols, fullNames, fullSector, fullIndustry  = GetAllSymbols()

    universe = pd.io.data.get_data_yahoo(fullSymbols[:], 
                                     start=datetime(2014, 9, 1), 
                                     end=datetime(2014, 10, 1))

    universe = universe.transpose(2,1,0)
    print universe.head
    print universe[fullSymbols[0]]
