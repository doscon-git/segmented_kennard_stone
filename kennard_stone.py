#%%
import numpy as np
import pandas as pd
from sklearn import metrics
import math
import psutil # to get the avilable memory on the machine

def kennardStone(data: pd.DataFrame,output_n: int ,verbose=False) -> np.array:
    
    """
    Calculates a Kennard-Stone sample selection.

        Parameters:
                data (DataFrame): The data to perform the selection on
                output_n (int): How many samples to select

        Returns:
                selected_indexes (array): a list containing the indexes that were selected
    """

    pairwise_distances = pd.DataFrame(metrics.pairwise_distances(data, metric="euclidean", n_jobs=-1))

    inital_selection = list(np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)) # Turns the flat index into a coordinate in the original matrix shape. Chooses row and colum that are the most different.
    
    if verbose:
        print("Initial selection {}, distance value {}".format(inital_selection,pairwise_distances.loc[inital_selection[0],inital_selection[1]]))

    # array to keep track of selected indexes.
    data_size = data.shape[0]
    selected = np.array([False]*data_size)

    # set the first selected points to true
    selected[inital_selection[0]] = True
    selected[inital_selection[1]] = True

    not_selected = ~selected

    n_counter = output_n # countdown until ouput_n reaches zero
    n_counter -= 2 # two points selected initially

    while n_counter > 0:
        # df.loc[row, col], row = not selected, col = selected
        row_mins = pairwise_distances.loc[not_selected,selected].min(axis=1)
        max_idx = row_mins.idxmax()
        selected[max_idx] = True
        not_selected[max_idx] = False
        n_counter -= 1

    return selected


def segmentedKennardStone(data: pd.DataFrame, output_n: int, verbose=False ,margin: float =0.7,force_memory=False):

    """
    Calculates a Kennard-Stone sample selection on a potentially large dataset. It will 
    attempt to split the dataset into multiple segments that can fit into memory and do 
    KennardStone selections on those subsets. It will then do a final Kennard-Stone
    selection on the combined subselections. 

        Parameters:
                data (DataFrame): The data to perform the selection on
                output_n (int): How many samples to select
                margin (float): Memory safety margin
                force_memory (bool or float): Force the use of a spesific memory size (GB) instead of calculating it.

        Returns:
                selected_indexes (array): a list containing the indexes that were selected
    """

    itemsize = data.to_numpy().itemsize
    input_n = data.shape[0]
    
    # Alternative: memory = psutil.swap_memory().free :  check swap vs. virtual memory
    if force_memory == False:
        memory = psutil.virtual_memory().available
    else:
        memory = force_memory*1024**3 # GB to bytes conversion
    max_segment_size = int((memory/itemsize)**0.5)
    initial_segment_size = int((memory*margin/itemsize)**0.5)
    
    segments_n = int(math.ceil(input_n/initial_segment_size)) # round up to be safe so pairwise won't be larger than memory)
    final_segment_size = int(input_n/segments_n)
    segment_output_n = int(output_n/segments_n + output_n/segments_n*(1-output_n/input_n))
    
    if verbose:
        print("Available memory is {:.2f}GB, itemsize is {} bytes".format(memory/(1024**3),itemsize))    
        print("Max items = {}. With margin at {} using segment size of {}".format(max_segment_size,margin,final_segment_size))
        print("Sub selection size calculated to {}".format(segment_output_n))
    
    first_selection = np.array([False]*input_n)
    for segment_n in range(0,segments_n):
        segment_start = final_segment_size*segment_n
        segment_end = final_segment_size*(segment_n+1)
        if verbose:
            print("Calculating Kennard-Stone for segment {} out of {}, indexes {}-{}".format(segment_n+1,segments_n,final_segment_size*segment_n,final_segment_size*(segment_n+1)))
        segment = data.iloc[segment_start:segment_end]

        from sklearn import metrics
        segment_ks = kennardStone(segment,segment_output_n,verbose=verbose)
        first_selection[segment_start:segment_end] = segment_ks
    
    if verbose:
        print("calculating final kennard stone on {} sub-selected samples, selecting {} samples".format(data[first_selection].shape[0],output_n))
    second_selection = kennardStone(data[first_selection],output_n,verbose=verbose)
    selected_indexes = data[first_selection][second_selection].index

    final_selection = np.array([False]*input_n)
    final_selection[selected_indexes] = True
    return final_selection


#%%
if __name__ == "__main__":
    np.random.seed(42)
    plotting = True
    data_size = 100
    output_n = 20
    """
    Non segmented
    """
    print("-"*10+"non segmented"+"-"*10)
    data = pd.DataFrame({"1":np.random.random(data_size)*5,"2":np.random.random(data_size)*10})
    selection = kennardStone(data,output_n,verbose=True)
    print("Selection split: [not selected, selected]",np.bincount(selection))

    # get the data itself
    selected_data = data[selection]
    not_selected_data = data[~selection]

    print(selected_data,"\n")
    # indexes as a list
    selected_indexes = np.where(selected_data)[0].tolist()

    # %%
    """
    Segmented with 1 segment
    """
    print("-"*10+"segmented with 1 segment"+"-"*10)
    selection = segmentedKennardStone(data,output_n,verbose=True)
    print("Selection split: [not selected, selected]",np.bincount(selection))
    
    # get the data itself
    selected_data = data[selection]
    not_selected_data = data[~selection]
    print(selected_data,"\n")

    # %%
    """
    Segmented selection with forced memory to ensure segmentation
    """
    print("-"*10+"segmented with 3 segments"+"-"*10)
    data_size = 8000
    output_n = 20
    data = pd.DataFrame({"1":np.random.random(data_size)*5,"2":np.random.random(data_size)*10})
    selection = segmentedKennardStone(data,output_n,verbose=True,force_memory=0.1)
    print("Selection split: [not selected, selected]",np.bincount(selection))

    selected_data = data[selection]
    not_selected_data = data[~selection]

    print(selected_data,"\n")

    # %%
    """
    Test on a large generated dataset. 
    Note: if you have more than 10 GB available this Will take a while...
    """
    extreme_test = False
    if extreme_test: # will only run if extreme test is enabled
        memory = psutil.virtual_memory().available
        output_n
        print("{:.2f} GB available, generating data to create segments".format(memory/(1024**3)))
        
        max_segment_size = int((memory/8)**0.5)
        data_size = max_segment_size * 5

        print("data size {}, expecting ~5 segments".format(data_size,5))

        data = pd.DataFrame({"1":np.random.random(data_size)*5,"2":np.random.random(data_size)*10})
        
        selection = segmentedKennardStone(data,output_n,verbose=True)
        selected_data = data[selection]
        not_selected_data = data[~selection]