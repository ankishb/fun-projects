
########################################################################
########################################################################

def get_groupby(df, col, groups):
    size = int(df.shape[0]//groups)
    gps = pd.DataFrame()
    rest_col = list(set(df.columns) - (set(col)))
    for s in range(groups):
        try:
            chunk = df[s*size : s*size + size]
        except:
            chunk = df[s*size : ]
        chunk_gp = chunk.groupby(col, as_index=False).count()
        if s == 0:
            gps = chunk_gp
            continue
        gps = gps.merge(chunk_gp, on=col, how='outer')
#         print(gps.head())
        gps_new = pd.DataFrame()
        gps_new[col] = gps[col]
        
        for c in rest_col:
            col_list = [c_ for c_ in gps.columns if c_.startswith(c)]
            gps_new[c] = gps[col_list].apply(lambda x: np.nansum(x), axis=1)
#         print(gps_new.head(), end="\n\n")
        gps = gps_new
        del gps_new
        gc.collect()
#         gps.columns = [col, 'col1', 'col2']
#         gps['new'] = gps[['col1','col2']].apply(lambda x: np.nansum(x), axis=1)
#         gps.drop(['col1','col2'], axis=1, inplace=True)
    return gps


########################################################################
########################################################################

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()

def process_image(img_file):
    img = cv2.imread(img_file)
    img = cv2.resize(img, sz).transpose((2,0,1)).astype('float32') / 255.0
    return img

In [4]:

start = time.time()

X_train = []
Y_train = []

for j in range(10):
    print('Load folder c{}'.format(j))
    path = os.path.join('../input/train', 'c' + str(j), '*.jpg')
    files = glob.glob(path)
    X_train.extend(Parallel(n_jobs=nprocs)(delayed(process_image)(im_file) for im_file in files))
    Y_train.extend([j]*len(files))
    
end = time.time() - start
print("Time: %.2f seconds" % end)




########################################################################
########################################################################
############### Generalize Parallel function syntax ################
def run_operation(args, file):
	# run_operation on files like
	img = cv2.imread(file)
	img = cv2.resize(file)
	save_file(img)
Parallel(n_jobs = num_cores)(delayed(run_operation)(args, file) for file in file_collections)



########################################################################
########################################################################



import multiprocessing as mp
import nltk

corpus = {f_id: nltk.corpus.gutenberg.raw(f_id)
          for f_id in nltk.corpus.gutenberg.fileids()}

def tokenize_and_pos_tag(pair):
    f_id, doc = pair
    return f_id, nltk.pos_tag(nltk.word_tokenize(doc))


if __name__ == '__main__':
    # automatically uses mp.cpu_count() as number of workers
    # mp.cpu_count() is 4 -> use 4 jobs
    with mp.Pool() as pool:
        tokens = pool.map(tokenize_and_pos_tag, corpus.items())




########################################################################
########################################################################



import numpy as np
from time import time

# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[200000, 5])
data = arr.tolist()


# Solution Without Paralleization

def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

results = []
for row in data:
    results.append(howmany_within_range(row, minimum=4, maximum=8))

print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]

"""
5. How to parallelize any function?

The general way to parallelize any operation is to take a particular function that should be run multiple times and make it run parallelly in different processors.

To do this, you initialize a Pool with n number of processors and pass the function you want to parallelize to one of Pools parallization methods.

multiprocessing.Pool() provides the apply(), map() and starmap() methods to make any function run in parallel.

Nice! So what’s the difference between apply() and map()?

Both apply and map take the function to be parallelized as the main argument. But the difference is, apply() takes an args argument that accepts the parameters passed to the ‘function-to-be-parallelized’ as an argument, whereas, map can take only one iterable as an argument.

So, map() is really more suitable for simpler iterable operations but does the job faster.

We will get to starmap() once we see how to parallelize howmany_within_range() function with apply() and map().
5.1. Parallelizing using Pool.apply()

Let’s parallelize the howmany_within_range() function using multiprocessing.Pool().
"""
# Parallelizing using Pool.apply()

import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]

# Step 3: Don't forget to close
pool.close()    

print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]

5.2. Parallelizing using Pool.map()

Pool.map() accepts only one iterable as argument. So as a workaround, I modify the howmany_within_range function by setting a default to the minimum and maximum parameters to create a new howmany_within_range_rowonly() function so it accetps only an iterable list of rows as input. I know this is not a nice usecase of map(), but it clearly shows how it differs from apply().

# Parallelizing using Pool.map()
import multiprocessing as mp

# Redefine, with only 1 mandatory argument.
def howmany_within_range_rowonly(row, minimum=4, maximum=8):
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

pool = mp.Pool(mp.cpu_count())

results = pool.map(howmany_within_range_rowonly, [row for row in data])

pool.close()

print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]

5.3. Parallelizing using Pool.starmap()

In previous example, we have to redefine howmany_within_range function to make couple of parameters to take default values. Using starmap(), you can avoid doing this. How you ask?

Like Pool.map(), Pool.starmap() also accepts only one iterable as argument, but in starmap(), each element in that iterable is also a iterable. You can to provide the arguments to the ‘function-to-be-parallelized’ in the same order in this inner iterable element, will in turn be unpacked during execution.

So effectively, Pool.starmap() is like a version of Pool.map() that accepts arguments.

# Parallelizing with Pool.starmap()
import multiprocessing as mp

pool = mp.Pool(mp.cpu_count())

results = pool.starmap(howmany_within_range, [(row, 4, 8) for row in data])

pool.close()

print(results[:10])
#> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]



########################################################################
########################################################################

The algorithm to do so is:
	1. Sort the data on the keys, on which grouping will be done.
    2. Load the next kkk rows from a dataset
    3. Identify the last group from the kkk rows
    4. Put the mmm rows corresponding to the last group aside (I call them orphans)
    5. Perform the groupby on the remaining k−mk - mk−m rows
    6. Repeat from step 1, and add the orphan rows at the top of the next chunk

For example let’s assume your data contains 42 gazillion rows (basically a lot of rows), sorted by the key attribute you want to do your groupby. You first start by reading the first kkk rows, say 10000. These kkk rows will contain a certain number of distinct key values (anything from 1 to kkk actually). The insight is that apart from the last key all the data for the rest of the keys is contained in the chunk. As for the last key there is some ambiguity. Indeed it could well be that there are rows in the next chunk that also belong to the last key. We thus have to put the rows that belong the last key aside because we’ll put them on top of the next chunk.

It turns out that this is pretty easy to do with pandas.

import pandas as pd

def stream_groupby_csv(path, key, agg, chunk_size=1e6):

    # Tell pandas to read the data in chunks
    chunks = pd.read_csv(p, chunksize=chunk_size)

    results = []
    orphans = pd.DataFrame()

    for chunk in chunks:

        # Add the previous orphans to the chunk
        chunk = pd.concat((orphans, chunk))

        # Determine which rows are orphans
        last_val = chunk[key].iloc[-1]
        is_orphan = chunk[key] == last_val

        # Put the new orphans aside
        chunk, orphans = chunk[~is_orphan], chunk[is_orphan]

        # Perform the aggregation and store the results
        result = agg(chunk)
        results.append(result)

    return pd.concat(results)

Let’s go through the code. We can use the chunksize parameter of the read_csv method to tell pandas to iterate through a CSV file in chunks of a given size. We’ll store the results from the groupby in a list of pandas.DataFrames which we’ll simply call results. The orphan rows are store in a pandas.DataFrame which is obviously empty at first. Every time we read a chunk we’ll start by concatenating it with the orphan rows. Then we’ll look at the last row of the chunk to determine the last key value. This will then allow us to put the new orphan rows aside and remove them from the chunk. Then we simply have to perform the groupby on the chunk and add the results to the list of results. The groupby happens in the agg function, which is provided by the user. The idea is to give the user the maximum amount of flexibility by letting provide the agg function. An example agg function could be:

agg = lambda chunk: chunk.groupby('passband')['flux'].mean()

You can even compute multiple aggregates on more than one field:

agg = lambda chunk: chunk.groupby('passband').agg({
    'flux': ['mean', 'std'],
    'mjd': ['min', 'max']
})


########################################################################
########################################################################



A natural thing we could next is to process the groupbys concurrently. Indeed we could use a worker pool to run the successive agg calls. Thankfully this is trivial to implement with Python’s multiprocessing module which is included in the default library.

import itertools
import multiprocessing as mp
import pandas as pd


def stream_groupby_csv(path, key, agg, chunk_size=1e6, pool=None, **kwargs):

    # Make sure path is a list
    if not isinstance(path, list):
        path = [path]

    # Chain the chunks
    kwargs['chunksize'] = chunk_size
    chunks = itertools.chain(*[
        pd.read_csv(p, **kwargs)
        for p in path
    ])

    results = []
    orphans = pd.DataFrame()

    for chunk in chunks:

        # Add the previous orphans to the chunk
        chunk = pd.concat((orphans, chunk))

        # Determine which rows are orphans
        last_val = chunk[key].iloc[-1]
        is_orphan = chunk[key] == last_val

        # Put the new orphans aside
        chunk, orphans = chunk[~is_orphan], chunk[is_orphan]

        # If a pool is provided then we use apply_async
        if pool:
            results.append(pool.apply_async(agg, args=(chunk,)))
        else:
            results.append(agg(chunk))

    # If a pool is used then we have to wait for the results
    if pool:
        results = [r.get() for r in results]

    return pd.concat(results)

I’ve added some extra sugar in addition to the new pool argument. The path argument can now be a list of strings, all of the listed will then we processed sequentially. This is pretty easy to do with itertools.chain method. The pool argument has to be a class with an apply_async method. I’ve implemented it so that it will work even you don’t provided any pool. From what I’m aware of the multiprocessing library has two different pool implementations, namely multiprocessing.Pool which uses processes and multiprocessing.pool.ThreadPool which uses threads. As a general rule of thumb threads are good for I/O bound tasks processes are good for CPU bound tasks.

To conclude here is an example of how to use the stream_groupby_csv method:

def agg(chunk):
    """lambdas can't be serialized so we need to use a function"""
    return chunk.groupby('some_sub_key')['some_column'].mean()

results = results = stream_groupby_csv(
    path=[
        'path/to/first/dataset.csv',
        'path/to/second/dataset.csv',
    ],
    key='some_key',
    agg=agg,
    chunk_size=100000,
    pool=mp.Pool(processes=8)
)