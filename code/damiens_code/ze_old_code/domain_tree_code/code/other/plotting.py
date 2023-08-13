# Import operating system
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Their imports from other packages
from jax.example_libraries import optimizers
import scipy.io
import jax
import jax.numpy as np

# My imports from other packages
import dill

# Imports from local files
from utils_fs_v2 import  DataGenerator, DataGenerator_res
from SFDomainNet_Class import DomainNet


if __name__=="__main__":

    with open("C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/dd_pinns/code/output/test", "rb") as fp:   # Unpickling
        NDTree = dill.load(fp)

    print(NDTree)
    # print("hello")
