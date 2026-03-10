import numpy as np
import pandas as pd
from scipy.optimize import minimize
from preprocessing import loading, ret


def portfolio_return(w: np.ndarray, exp: pd.Series) -> float:
    #compute expected portfolio return
    #factor in Sharpe focrmula

    n = float(np.dot(w, exp))

    # simply the weighted average of the asset returns
    # expected = mean of each coin

    # Rp​=w1​r1​+w2​r2​+w3​r3​+...wn​rn​

    return n


def portfolio_volatility(w: np.ndarray, cov: pd.DataFrame) -> float:
    #compute portfolio volatility.

    #factor in Sharpe focrmula
    # σp​ = √  w    Σ   w
    # wT = transposé weigths                    Σ = covariance matrix
    
    
    

    n= float(   np.sqrt(np.dot(w , np.dot(cov, w)))    )
    
    return n




def negative_sharpe_ratio( weights: np.ndarray,  expected_returns: np.ndarray, cov_matrix: np.ndarray) -> float:
    
    #since scipy.optimize.minimize() can only minimize, you convert the Sharpe ratio maximization problem
    #into a minimization problem by returning the negative Sharpe ratio

    #Maximizing Sharpe = minimizing  -Sharpe
    
    
    #Negative Sharpe ratio for minimization.
    
    p_return = portfolio_return(weights, expected_returns) #Rp
    p_vol = portfolio_volatility(weights, cov_matrix) #σp​

    if p_vol <= 1e-12:      # si vol devient extremely small: _/0 = inf
        return np.inf

    sharpe = p_return  / p_vol  # on neglige risk free rate.

    return -sharpe








def optimise( returns: pd.DataFrame,  max_weight: float = 0.7, annualize: bool = True) -> pd.Series:
  

    # returns weigths optimaux in panda series

    if returns.empty:
        raise ValueError("Returns DataFrame is empty.") #make sure input not emtpy 
    

    n_assets = returns.shape[1]     #nombre de cryptos = 5


    if max_weight * n_assets < 1:           # raises error if max_weigth contraint is too low: at certain point it wouldn't be able to reach 100% (the first constraint)
        raise ValueError( " constraint pas possible: max_weight is too low for the number of assets" )   #for us no problem since by def = 70%




    expected_returns = returns.mean()       #logic defined earlier
    cov_matrix = returns.cov()




    # Annualize if desired
    if annualize:
        expected_returns = expected_returns * 252
        cov_matrix = cov_matrix * 252

    expected_returns = expected_returns.values
    cov_matrix = cov_matrix.values




    # initial guess: equal weights
    initial_weights = np.ones(n_assets) / n_assets

    bounds = [(0.0, max_weight) for _ in range(n_assets)] #chaque weigth inclue entre [0 et 0.7] (no short selling)

    # Constraint: fully invested portfolio
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        #"type": "eq" = whatever the function returns must equal 0
        # np.sum(w)−1=0
        # <==> np.sum(w)=1 

        # la somme des weigths = 100%               = first constraint: whole portfolio invested
    ]


    #minimzing -Sharpe
    result = minimize(
        fun=negative_sharpe_ratio,  #le sharpe neg
        x0=initial_weights,     #equal distribution de départ
        args=(expected_returns, cov_matrix),    #arguments used
        method="SLSQP",
        bounds=bounds,      #2e constraint: les bornes pour chacun des weigths
        constraints=constraints,        #1er constraint: la somme des poids = 1
        options={"maxiter": 1000, "ftol": 1e-9}     #precision parameters (n itteration, tolerance precision)
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    optimal= pd.Series(result.x, index=returns.columns, name="optimal weights")
    

    return optimal





def optreturns(returns: pd.DataFrame, weights: pd.Series ) -> pd.Series:

    # compute optimized portfolio return series using optimal weigths

    return returns @ weights    #multiplication matricielle





if __name__ == "__main__":          # to only run if called from here
    prices = loading("closing.csv")
    returns = ret(prices)

    weights = optimise(
        returns,
        max_weight=0.7,
        annualize=True
    )

    portfolio_returns = optreturns(returns, weights)

    print("\nOptimal Weights:\n")
    print(weights)

    print("\nSum of weights:")
    print(weights.sum())

    print("\nOptimized Portfolio Returns (first 5 rows):\n")
    print(portfolio_returns.head())


