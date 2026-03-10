import numpy as np
import pandas as pd


def monte_carlo_portfolio_simulation( returns: pd.DataFrame, weights: pd.Series, n_simulations: int = 5000, horizon: int = 252, annualize: bool = False ) -> np.ndarray:
    

    # simulates the final value of a portfolio after a chosen future time period:
    # if I simulate the future many times, what could my portfolio be worth at the end?

    # can vary number of simulations which affects precision (how many possible futures to generate)
    # also horizon (time window of simulation)



    if returns.empty:       #no data error
        raise ValueError("Returns DataFrame is empty.")

    if not all(col in returns.columns for col in weights.index):        #ensures no misimatch of column order
        raise ValueError("Weights index must match returns columns.")


    weights = weights.reindex(returns.columns).values       #ensures no misimatch of column order
    mean_vector = returns.mean().values #expected return of each asset
    cov_matrix = returns.cov().values 

    if annualize:       #scaling if wanted annual
        mean_vector = mean_vector / 252
        cov_matrix = cov_matrix / 252


    terminal_values = np.empty(n_simulations)       #will store results from each simulation

    for i in range(n_simulations):
        simm = np.random.multivariate_normal( mean=mean_vector, cov=cov_matrix, size=horizon ) # input calculated parameters

        retu = simm @ weights    #apply weigths

        terminal_values[i] =  np.prod(1 + retu)

    return terminal_values  #last value of cumulative return for each sim




def monte_carlo_portfolio_paths( returns: pd.DataFrame, weights: pd.Series, n_simulations: int = 1000, horizon: int = 252,) -> np.ndarray:
    
    # simulates the entire path of portfolio values through time, not just the final value
  

    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")     #same
    weights = weights.reindex(returns.columns).values       #same
    mean_vector = returns.mean().values
    cov_matrix = returns.cov().values




    paths = np.empty((horizon + 1, n_simulations))      #storage for values 
    paths[0, :] = 1     # initialisé à 1 kolohom


    for i in range(n_simulations):
        simm = np.random.multivariate_normal( mean=mean_vector, cov=cov_matrix, size=horizon  ) #same

        retu = simm @ weights
        paths[1:, i] =   np.cumprod(1 + retu)

    return paths        # matrix returend: rows= days of time horrizon; columns: simulations
#for each day and simulation computes porrtfolio value that day

'''
ex:

    Day	Sim1	Sim2	Sim3
    0	1.00	1.00	1.00
    1	1.01	0.99	1.03

'''




import matplotlib.pyplot as plt
import numpy as np


def plot_terminal_value_distribution(
    control_terminal: np.ndarray,
    optimized_terminal: np.ndarray
) -> None:
    """
    Plot histograms of simulated terminal portfolio values.
    """

    plt.figure(figsize=(10, 6))
    plt.hist(control_terminal, bins=50, alpha=0.6, label="Control Portfolio")
    plt.hist(optimized_terminal, bins=50, alpha=0.6, label="Optimized Portfolio")
    plt.title("Monte Carlo Distribution of Terminal Portfolio Values")
    plt.xlabel("Terminal Portfolio Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt
import numpy as np


def plot_monte_carlo_paths(
    paths: np.ndarray,
    title: str = "Monte Carlo Portfolio Paths"
) -> None:
    """
    Plot a sample of simulated portfolio paths.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(paths[:, :100], alpha=0.3)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()