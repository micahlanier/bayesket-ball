### Setup

# Libraries.
import numpy as np
import pymc

### Main Functionality

def model_games (
    data,
    features=['team_Pythag'],
    coef_dists=None,
    b0_params={'mu':0, 'tau':0.0003, 'value':0},
    err_params=[.5],
    default_coef_dist = pymc.Normal,
    default_coef_params = {'mu':0, 'tau':0.0003, 'value':0},
    step_method = pymc.Metropolis,
    step_method_params = None,
    default_step_method_params = {'proposal_sd':1., 'proposal_distribution':'Normal'}
):
    """
    Inputs:
        data: Pandas dataframe of game information.
        features: String list of features from game data.
        coef_dists: Distribution for coefficients. Default: see default_coef_dist; usually Normal.
        b0_params: Parameters for intercept ("b0"). Default: mean=0, precision=0.0003; initial value=0.
        err_params: Parameters list for Bernoulli-distribued error distribution. Default: p=0.5.
        default_coef_dist: Coefficient distributions.
        default_coef_params: Default coefficient distribution parameters.
        step_method: MCMC stepping method.
        step_method_params: Parameters for stepping method for each coefficient. Default: see default_step_method_params.
        default_step_method_params: Default step method parameters for coefficient draws.
    Returns:
        PyMC MCMC object. Call with .sample() and desired parameters to perform actual sampling.
    """
    
    # Define priors on intercept and error. PyMC uses precision (inverse variance).
    b0 = pymc.Normal('b_0', **b0_params)
    err = pymc.Bernoulli('err', *err_params)
    
    # Containers for coefficients and data.
    b = np.empty(len(features), dtype=object)
    x = np.empty(len(features), dtype=object)
    
    # Traverse features.
    for i, f in enumerate(features):
        # Coefficient.
        # First start with the distribution. Use one if we've been given one; else use default.
        coef_dist_type = default_coef_dist if coef_dists is None or coef_dists[i] is None or coef_dists[i][0] is None else coef_dists[i][0]
        # Now handle parameters.
        coef_dist_params = default_coef_params if coef_dists is None or coef_dists[i] is None or coef_dists[i][1] is None else coef_dists[i][1]
        # Now actually create the coefficient distribution
        b[i] = coef_dist_type('b_'+f, **coef_dist_params)
        # Data distribution.
        x[i] = pymc.Normal('x_'+f, 0, 1, value=np.array(data[f]), observed=True)
    
    # Logistic function.
    @pymc.deterministic
    def logistic(b0=b0, b=b, x=x):
        return 1.0 / (1. + np.exp(-(b0 + b.dot(x))))

    # Get outcome data.
    y = np.array(data.win)
    # Model outcome as a Bernoulli distribution.
    y = pymc.Bernoulli('win', logistic, value=y, observed=True)
    
    # Define model, MCMC object.
    model = pymc.Model([logistic, pymc.Container(b), err, pymc.Container(x), y])
    mcmc  = pymc.MCMC(model)
    
    # Configure step methods.
    for var in list(b)+[b0,err]:
        coef_step_method_params = default_step_method_params if step_method_params is None or step_method_params[i] is None else step_method_params[i]
        mcmc.use_step_method(step_method, stochastic=var, **default_step_method_params)
    
    # Return MCMC object.
    return mcmc

### Standalone Run

# Code to run if run stand-alone.
if __name__ == '__main__':
    main()
def main():
    pass
