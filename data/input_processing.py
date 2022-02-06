import pandas as pd
import numpy as np


class RawData_index:

    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end

    # Initial dataframe to find indexes of first and last datapoint
    def dfinit(self):
        df = pd.read_csv(self.name, skiprows=1, on_bad_lines='skip', usecols=['TIMESTAMP'])
        #print(df)
        first = -1
        last = -1
        lt = len(self.start)

        df = df.to_numpy()
        for i in range(len(df)):
            if str(df[i][0])[0:lt] == self.start:
                first = i
            if str(df[i][0])[0:lt] == self.end:
                last = i

        if (first == -1) or (last == -1):
            print('ERROR! Timestamp does not work!')
            return

        return first, last

    # finding index of columns
    def cols(self):
        df = pd.read_csv(self.name, skiprows=1, low_memory=False)

        ti1 = df.columns.get_loc('T11')
        te1 = df.columns.get_loc('T21')
        ti2 = df.columns.get_loc('DT1temp')
        te2 = df.columns.get_loc('DT2temp')
        hf1 = df.columns.get_loc('HF1')
        hf2 = df.columns.get_loc('HF2')

        return [ti1, te1, ti2, te2, hf1, hf2]

# Extracting selected vectors from raw data file TRSYS01_Public.dat
class RawData_series:

    def __init__(self, name, f_ind, l_ind, columns):
        self.name = name
        self.f_ind = f_ind
        self.l_ind = l_ind
        self.columns = columns

    # Extracting function and averaging for every "ts_avg" minutes
    def ex_vect(self):

        first = self.f_ind + 2
        num_rows = self.l_ind - self.f_ind

        x_val = []

        for i in range(num_rows):
            x_val.append(i)

        x_val = np.array(x_val)

        df = pd.read_csv(self.name, usecols=self.columns, skiprows=first,
                        nrows=num_rows, header=None)

        for col in range(len(self.columns)):
            globals()[f"c{col}"] = df.iloc[:,col].to_numpy()

        return [c0, c1, c2, c3, c4*(-1), c5*(-1), x_val]