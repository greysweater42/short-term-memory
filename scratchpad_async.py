# import asyncio
# import pandas as pd
# from src.filter_out_noise import filter_out_noise
from src.utils import timeit


import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


# class RawPerson:
#     def __init__(self) -> None:
#         self.eeg = pd.read_csv("eeg.csv", index_col=0)

#     @timeit
#     def apply_filters(self):
#         results = asyncio.run(self.fourier_transform_coroutine())  # run coroutine
#         self.eeg = pd.concat(results, axis=1)

#     async def fourier_transform_coroutine(self):
#         # aws - short for awaitables
#         aws = [asyncio.create_task(filter_out_noise(self.eeg[channel])) for channel in self.eeg.columns]
#         return await asyncio.gather(*aws)


# raw_person = RawPerson()
# raw_person.apply_filters()
# print(raw_person.eeg.head())

import asyncio
from datetime import datetime
import pandas as pd


# async def filter_out_noise(i):
#     await asyncio.sleep(i)  # asyncio strongly prefers asyncio.sleep over time.sleep
#     print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {i}")
#     return i
# %% not working
# from scipy.fft import fft, fftfreq, ifft
# import numpy as np
# import pandas as pd


# # TODO move to FourierTransformer
# async def filter_out_noise(eeg: pd.Series) -> pd.Series:
#     """filters out unwanted frequencies;
#     some current frequencies are typical for electricity providers and they add noise to the data;
#     this function filters them out. Also high-pass and low-pass filters are applied on frequencies
#     for which brainwaves do not occur"""
#     freq = fftfreq(len(eeg), 0.002)  # 500Hz -> 0.002s (2ms)
#     # y_fft = fft(eeg.to_numpy()).real
#     eeg = pd.read_csv("eeg.csv", index_col=0)
#     # # 50Hz line filter
#     # y_fft[(np.abs(freq) > 45) & (np.abs(freq) < 51)] = 0
#     # y_fft[(np.abs(freq) > 99) & (np.abs(freq) < 101)] = 0
#     # y_fft[(np.abs(freq) > 149) & (np.abs(freq) < 151)] = 0
#     # # low-pass filter
#     # y_fft[(np.abs(freq) > 45)] = 0
#     # # high-pass filter
#     # y_fft[(np.abs(freq) < 2)] = 0
#     # return pd.Series(ifft(y_fft).real, index=eeg.index, name=eeg.name)
#     # await asyncio.sleep(1)  # asyncio strongly prefers asyncio.sleep over time.sleep
#     print(f"{datetime.now().strftime('%H:%M:%S.%f')}")
#     return eeg


# class RawPerson:
#     def __init__(self) -> None:
#         self.eeg = pd.read_csv("eeg.csv", index_col=0)

#     async def coro(self):
#         # aws - short for awaitables
#         aws = [asyncio.create_task(filter_out_noise(self.eeg[channel])) for channel in self.eeg.columns[:2]]
#         return await asyncio.gather(*aws)
#         # aws = [asyncio.create_task(filter_out_noise(self.eeg[channel])) for channel in self.eeg.columns]
#         # return await asyncio.gather(*aws)

#     @timeit
#     def run(self):
#         results = asyncio.run(self.coro())  # run coroutine
#         # print(results)

# raw_person = RawPerson()
# raw_person.run()

# %% works
# class RawPerson:
#     def __init__(self) -> None:
#         self.eeg = pd.read_csv("eeg.csv", index_col=0)

#     async def sleep_int(self, i):
#         await asyncio.sleep(i)  # asyncio strongly prefers asyncio.sleep over time.sleep
#         print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {i}")
#         return i

#     async def coro(self):
#         # aws - short for awaitables
#         aws = [asyncio.create_task(self.sleep_int(i)) for i in range(5)]
#         return await asyncio.gather(*aws)

#     @timeit
#     def run(self):
#         results = asyncio.run(self.coro())  # run coroutine
#         print(results)


# raw_person = RawPerson()
# raw_person.run()


# %% works
# import asyncio
# from datetime import datetime


# class RawPerson:
#     def __init__(self) -> None:
#         pass

#     async def sleep_int(self, i):
#         await asyncio.sleep(i)  # asyncio strongly prefers asyncio.sleep over time.sleep
#         print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {i}")
#         return i

#     async def coro(self):
#         # aws - short for awaitables
#         aws = [asyncio.create_task(self.sleep_int(i)) for i in range(5)]
#         return await asyncio.gather(*aws)

#     def run(self):
#         results = asyncio.run(self.coro())  # run coroutine
#         print(results)

# raw_person = RawPerson()
# raw_person.run()


# import asyncio
# from datetime import datetime


# async def sleep_int(i):
#     await asyncio.sleep(i)  # asyncio strongly prefers asyncio.sleep over time.sleep
#     eeg = pd.read_csv("eeg.csv", index_col=0)
#     print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {i}")
#     return i


# async def coro():
#     # aws - short for awaitables
#     aws = [asyncio.create_task(sleep_int(i)) for i in range(5)]
#     return await asyncio.gather(*aws)


# results = asyncio.run(coro())  # run coroutine
# print(results)


import time
from datetime import datetime
from multiprocessing import Pool


def sleep_int(i):
    # time.sleep(3)
    eeg = pd.read_csv("eeg.csv", index_col=0)
    return i


with Pool(3) as executor:
    print(datetime.now().strftime('%H:%M:%S.%f'))
    results = executor.map(sleep_int, range(3))
    print(type(results))
    for result in results:
        print(f"{datetime.now().strftime('%H:%M:%S.%f')}: {result}")