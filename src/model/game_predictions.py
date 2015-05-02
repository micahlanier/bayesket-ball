### Setup

# Libraries.
import bayes_lr
import numpy as np
import pandas as pd
import pymc

### Main Functionality

def logistic (s):
    """
    """
    return 1 / (1+np.exp(-s))

def predict_games (data, features, model_mcmc=None, coefs=None, method='map'):
    """
    Inputs:
        data:       Dataframe with game data.
        features:   List of features in data dataframe.
        model_mcmc: PyMC MCMC object. Required if 'coefs' not supplied.
        coefs:      Numpy array of coefficients; each column corresponds to an element of 'features'. Required if 'model_mcmc' is not supplied.
        method:     'pp' for posterior predictive, 'map' for single estimate using MAP.
    Returns:
        Binary predictions for each game.
        Raw continuous predictions (in range [0,1]) for each game.
        Accuracy based on binary predictions (range [0,1]).
    """

    # Trace length.
    coef_samples = model_mcmc.trace('b_0')[:].shape[0] if coefs is None else len(coefs)
    
    # Get coefficients.
    if coefs is None:
        coefs = bayes_lr.feature_coefficients(model_mcmc, features)
    # Use MAP if we need to.
    if method == 'map' and coefs.shape[0] > 1:
        coefs = coefs.mean(axis=0).reshape((1,len(features)+1))
        coef_samples = 1
    
    # Get design matrix.
    X = np.concatenate((np.ones((len(data),1)), data[features]), axis=1)
    
    # Estimate wins. Start with a wins container.
    y_hat_raw = np.zeros((coef_samples,len(data)))
    # Traverse set of coefficients and assemble win estimates.
    for c_i, c in enumerate(coefs):
        y_hat_raw[c_i] = logistic((c.T*X).sum(axis=1))
    # Tag wins/losses with comparison.
    y_hat = (y_hat_raw >= .5).astype(np.int)
    
    # Compute accuracy by wins if they are available.
    y_hat_accuracy = None
    if 'win' in data:
        y_hat_accuracy = (np.array(data.win) == y_hat).mean()
    
    return y_hat_raw, y_hat, y_hat_accuracy

# Main function for simulating a tournament.
def simulate_tournament (bracket, team_stats, features, model_mcmc=None, coef_trace=None):
    """
    Simulates a tournament, following all teams down the bracket to the championship.
    Inputs:
        bracket:    List of first round matchups. Should be first round of the tournament with the bracket "unrolled".
                    That is, the winner of game 1 will play the winner of game 2; 3 vs. 4; etc.
        team_stats: Dataframe with team statistics.
        features:   List of features in data dataframe.
        model_mcmc: PyMC MCMC object. Trace length = simulations. Required if 'coef_trace' not supplied.
        coef_trace: Numpy array of coefficients; each column corresponds to an element of 'features'. Trace length = simulations. Required if 'model_mcmc' is not supplied.
    Returns:
        TODO
    """

    ### Setup

    # Argument assertions.
    assert model_mcmc is not None or coef_trace is not None

    # Get coefficients if we were not given them.
    if coef_trace is None:
        coef_trace = bayes_lr.feature_coefficients(model_mcmc, features)

    ### Simulations

    # Get sorted teams.
    teams = np.array(bracket).ravel()
    teams.sort()
    # Use a numpy array to hold our outcomes.
    outcomes = np.zeros((len(teams),int(np.log2(len(teams)))))

    # Iterate over trace and start one branch per set of coefficients.
    for c in coef_trace:
        # Ensure correct shape for coefficients.
        c = c.reshape((1,c.shape[0]))
        # Simulate.
        c_outcome = simulate_bracket_coefs(bracket, team_stats, features, c)
        # Add simulation outcomes to main outcomes array.
        for t_i, t in enumerate(teams):
            outcomes[t_i,:] += c_outcome[t]

    # Turn team list and outcomes into a data frame.
    tournament_outcomes = pd.DataFrame(outcomes, index=teams, columns=['wins_round_'+str(rnd) for rnd in xrange(1,outcomes.shape[1]+1)])

    # Return teams and outcomes.
    return tournament_outcomes

# Helper function for simulating a bracket for one set of coefficients.
def simulate_bracket_coefs (bracket, team_stats, features, coefs):
    """
    """

    ### Data

    # Start by randomly shuffling each pair of teams.
    shuffled_bracket = []
    for game in bracket:
        flip = np.random.randint(2)
        shuffled_bracket.append([game[flip],game[np.abs(flip-1)]])

    # Create DF.
    games = pd.DataFrame(shuffled_bracket, columns=['team','opponent'])

    # Also retain an array representation of games for checking winners later.
    games_arr = np.array(games)

    # Append presumed accurate location columns. Assume all locations neutral.
    games['location_Neutral'] = 1
    for c in ['location_Home','location_SemiAway','location_SemiHome']:
        games[c] = 0

    # Trim team stats to only the teams represented.
    team_stats = team_stats[(team_stats.TeamName.isin(games.team)) | (team_stats.TeamName.isin(games.opponent))]

    # Merge with KenPom data.
    # Merge teams.
    games = games.merge(team_stats, left_on='team', right_on='TeamName')
    games.drop('TeamName', axis=1, inplace=True)
    games.columns = ['team_'+c if (c in team_stats.columns and c != 'year') else c for c in games.columns]
    # Merge opponents.
    games = games.merge(team_stats, left_on='opponent', right_on='TeamName')
    games.drop('TeamName', axis=1, inplace=True)
    games.columns = ['opponent_'+c if (c in team_stats.columns and c != 'year') else c for c in games.columns]

    # Calculate aggregate columns
    # Get all team, opponent columns.
    team_cols     = [c for c in games.columns if c.startswith('team_')]
    opponent_cols = [c for c in games.columns if c.startswith('opponent_')]
    # Calculate diff and ratio col names.
    diff_col_names  = [c.replace('team_','diff_')  for c in team_cols]
    ratio_col_names = [c.replace('team_','ratio_') for c in team_cols]
    # Calculate differences and ratios.
    diff_vals  = np.array(games.ix[:,team_cols]) - np.array(games.ix[:,opponent_cols])
    ratio_vals = np.array(games.ix[:,team_cols]) / np.array(games.ix[:,opponent_cols])
    # Convert to DF.
    diff_vals_df  = pd.DataFrame(diff_vals,  columns=diff_col_names)
    ratio_vals_df = pd.DataFrame(ratio_vals, columns=ratio_col_names)
    # Append to game features.
    games = pd.concat((games,diff_vals_df),  axis=1)
    games = pd.concat((games,ratio_vals_df), axis=1)

    ### Simulations

    # Simulate games.
    _1, y_hat, _2 = predict_games(data=games, features=features, coefs=coefs, method='map')

    # Manipulate win/loss data for indexing.
    winners_losers = y_hat.reshape((y_hat.shape[1],1))
    winners_losers = np.concatenate((winners_losers,(np.abs(winners_losers-1))), axis=1)
    winners_losers = winners_losers.ravel()
    # Get winners/losers.
    winners = games_arr.ravel()[winners_losers == 1]
    losers  = games_arr.ravel()[winners_losers == 0]

    # Construct dictionary of results.
    results = dict((w,[1]) for w in winners)
    results.update(dict((l,[0]*int(np.log2(len(bracket*2)))) for l in losers))

    # Different behavior for tournament end.
    if len(winners) == 1:
        # We just simulated the championship. Return the outcome.
        return results
    else:
        # There are still further rounds to go.
        # Calculate next bracket.
        next_bracket = winners.reshape((len(winners)/2,2)).tolist()
        # Simulate.
        next_results = simulate_bracket_coefs(next_bracket, team_stats, features, coefs)
        # Update our results and return.
        for t in next_results.keys():
            results[t] = results[t]+next_results[t]
        return results

### Standalone Run

# Code to run if run stand-alone.
if __name__ == '__main__':
    main()
def main():
    pass