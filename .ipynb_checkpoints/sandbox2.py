import numpy as np
import pandas as pd
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

df = pd.DataFrame(np.random.randint(3, 10, size=[500, 2]))
print(df)
def func(df):
    return df.shape

cores=mp.cpu_count()

df_split = np.array_split(df, cores, axis=0)

# create the multiprocessing pool
pool = Pool(cores)

# process the DataFrame by mapping function to each df across the pool
df_out = np.vstack(pool.map(func, df_split))
print(df_out)
# close down the pool and join
pool.close()
pool.join()
pool.clear()
