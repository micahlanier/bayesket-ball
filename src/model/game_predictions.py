### Setup

# Libraries.
import numpy as np
import pymc

### Main Functionality

def predict_games (data, model_mcmc, features, method='map'):
    """
    Inputs:
        data:       Dataframe with game data.
        model_mcmc: PyMC MCMC object.
        features:   list of features in data dataframe.
        method:     'pp' for posterior predictive, 'map' for single estimate using MAP.
    Returns:
        Binary predictions for each game.
        Raw continuous predictions (in range [0,1]) for each game.
        Accuracy based on binary predictions (range [0,1]).
    """

    # Trace and feature length.
    trace_len = model_mcmc.trace('b_0')[:].shape[0]
    coef_samples = trace_len
    feat_len  = len(features)+1
    data_len  = len(data)
    
    # Get coefficients.
    coefs = np.empty((trace_len,feat_len))
    for f_i, f in enumerate(['0']+features):
        coefs[:,f_i] = model_mcmc.trace('b_'+f)[:]
    # Use MAP if we need to.
    if method == 'map':
        coefs = coefs.mean(axis=0).reshape((1,feat_len))
        coef_samples = 1
    
    # Get design matrix.
    X = np.ones((data_len,feat_len))
    for f_i, f in enumerate(features):
        X[:,f_i+1] = data.ix[:,f]
    # Get wins.
    y = np.array(data.win)
    
    # Logistic function for win calculation.
    logistic = lambda s: 1 / (1+np.exp(-s))
    
    # Estimate wins. Start with a wins container.
    y_hat_raw = np.zeros((coef_samples,data_len))
    y_hat     = np.zeros((coef_samples,data_len), dtype=np.int)
    # Traverse set of coefficients and assemble win estimates.
    for c_i, c in enumerate(coefs):
        y_hat_raw[c_i] = logistic((c.T*X).sum(axis=1))
    # Tag wins/losses with comparison.
    y_hat = (y_hat_raw >= .5).astype(np.int)
    
    # Compute accuracy by wins.
    y_hat_accuracy = (y == y_hat).mean()
    
    return y_hat_raw, y_hat, y_hat_accuracy

### Standalone Run

# Code to run if run stand-alone.
if __name__ == '__main__':
    main()
def main():
    pass
