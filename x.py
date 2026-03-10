from preprocessing import ret, loading
import numpy as np
import pandas as pd




def control_portfolio(returns: pd.DataFrame):

    # The most common benchmark in portfolio research is: euqal allocation == each asset = 1/6


    n_ass = returns.shape[1] # number of columns excluding index = number of assets

    weights = np.ones(n_ass) / n_ass # weigth for each = 1/6
    weights = pd.Series(weights, index=returns.columns) # label weigths b esm el columns of returns


    portfolio_returns = returns.dot(weights) # weigthed sum == each coin his weigth     THEN    sum each row to get total up/down per day

    return weights, portfolio_returns



prices = loading("closing.csv")
returns = ret(prices)
w,p= control_portfolio( returns)
print (p)
print(returns.head)
