import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import loading, ret


def summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:

   # Compute descriptive statistics for each asset return series.

    stats = pd.DataFrame({
        "Mean": returns.mean(),
        "Median": returns.median(),
        "Std Dev": returns.std(),
        "Variance": returns.var(),
        "Min": returns.min(),
        "Max": returns.max(),
        "Skewness": returns.skew(),
        "Kurtosis": returns.kurt()
    })
    
    #each of the genertaed is a pandas series with a value for each column/ asset

    return stats

#prices = loading("closing.csv")
#returns = ret(prices)
#stats = summary_statistics(returns)
#print(stats)





def matrices(returns: pd.DataFrame) -> pd.DataFrame:

    # computes the correlation and covariance matrix of asset returns.

    # correlation for diversification and covariance to input into optimizer later 

    cor= returns.corr()
    cov = returns.cov()

    return cor, cov

def plotcor(corr:pd.DataFrame) :
    
    # plot seaborn heatmap of the correlation matrix.
    
    plt.figure(figsize=(8, 6))
    sns.heatmap( corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5 )

    plt.title("crypto return correlation matrix")
    plt.tight_layout()
    plt.show()

prices = loading("closing.csv")
returns = ret(prices)
cor, cov = matrices(returns)
print (cov)
#plotcor(cor)





def cumulative_return(returns: pd.Series) -> float: #the total return over the entire sample period.

    # (1 + returns) to make it a growth factor: 0.2 --> 1.02 
    # prod multiplies all the factors == the finale result is how much the investment became of the initial value

    return (1 + returns).prod() - 1



def drawdown_series(returns: pd.Series) -> pd.Series:
    # drawdown measures how far an investment has fallen from its previous peak
    wealth =  (1 + returns).cumprod()       #same as cumulative_return() output but keeps in series

    running_max = wealth.cummax()           #at every point it records highest value seen so far

    drawdown = (wealth - running_max) / running_max
    return drawdown     #panda series bardo


def maximum_drawdown(returns: pd.Series) -> float:      # the worst drop experienced
    dd = drawdown_series(returns)
    return dd.min()     #min car dd are negative numbers == absolute max




def annualized_return(returns: pd.Series) -> float:    # converts daily returns into a comparable yearly performance measure

    compounded_growth = cumulative_return(returns) +1
    n_periods = len(returns)        #number of observation days


    return compounded_growth ** (252 / n_periods) - 1


def annualized_volatility(returns: pd.Series) -> float:  #the standard deviation of returns scaled to a yearly basis


    return returns.std() * np.sqrt(252)    #same formula of iv


def sharpe_ratio(returns: pd.Series) -> float: #measures the return earned per unit of risk

    ann_return = annualized_return(returns)
    ann_vol = annualized_volatility(returns)

    if ann_vol <= 1e-12:
        return np.nan

    return (ann_return) / ann_vol      #same logic as function in optim.py, different arguments





def compare_portfolios(control_returns: pd.Series, optimized_returns: pd.Series) -> pd.DataFrame:

    control_stats = pd.Series({
        "annualized return": annualized_return(control_returns),
        "annualized volatility": annualized_volatility(control_returns),
        "sharpe ratio": sharpe_ratio(control_returns),
        "cumulative return": cumulative_return(control_returns),
        "maximum drawdown": maximum_drawdown(control_returns)
    })

    optimized_stats = pd.Series({
        "annualized return": annualized_return(optimized_returns),
        "annualized volatility": annualized_volatility(optimized_returns),
        "sharpe ratio": sharpe_ratio(optimized_returns),
        "cumulative return": cumulative_return(optimized_returns),
        "maximum drawdown": maximum_drawdown(optimized_returns)
    })

    comparison = pd.DataFrame({
        "control portfolio": control_stats,
        "optimized portfolio": optimized_stats
    })

    return comparison


def plot_cum(  control_returns: pd.Series,  optimized_returns: pd.Series) -> None:

    #to plot cumulative visual in comparaison

    control_curve =  (1 + control_returns).cumprod()
    optimized_curve = (1 + optimized_returns).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(control_curve, label="Control Portfolio", linewidth=2)
    plt.plot(optimized_curve, label="Optimized Portfolio", linewidth=2)
    plt.title("Cumulative Portfolio Performance")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()