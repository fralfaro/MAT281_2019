# metrics 

import numpy as np 

# a) Scale-dependent errors

def mae(targets,predictions):
    error = predictions - targets
    return np.abs(error).mean()

def mse(targets,predictions):
    error = predictions - targets
    return np.sqrt((error ** 2).mean())

# b) Percentage errors

def mape(targets,predictions):
    error = predictions - targets
    
    if any(x == 0 for x in targets):
        return np.inf
    else:
        return np.abs(error/targets).mean()
        

def smape(targets,predictions):
    error = predictions - targets
    sum_values = np.abs(predictions)+np.abs(targets)
    
    if any(x == 0 for x in sum_values):
        return np.inf
    
    else:
        return np.mean(np.abs(error)/sum_values)