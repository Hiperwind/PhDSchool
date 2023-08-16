try:

    import numpy as np
    import scipy.stats as stats
    import scipy.optimize
    import matplotlib.pyplot as plt

    import chaospy as cp 

    import pandas as pd
    import openturns as ot
    import copulogram as cp 

    import otkerneldesign as otkd

    print("Success: all packages are correctly installed")

except:
    print("Error: a package is missing")