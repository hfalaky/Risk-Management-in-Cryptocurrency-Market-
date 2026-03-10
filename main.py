from preprocessing import loading, ret
from x import control_portfolio
from stats import summary_statistics, matrices, plotcor, compare_portfolios, plot_cum
from optim import optimise, optreturns


def main():
    # Load and preprocess data
    prices = loading("closing.csv")
    returns = ret(prices)

    # Asset-level statistics
    stats = summary_statistics(returns)
    corr, cov = matrices(returns)

    # Control portfolio
    control_weights, control_returns = control_portfolio(returns)

    # Optimized portfolio
    optimal_weights = optimise(
        returns,
        max_weight=0.7,
        annualize=True
    )
    optimized_returns = optreturns(returns, optimal_weights)

    # Portfolio comparison table
    comparison = compare_portfolios(control_returns, optimized_returns)

    # Output
    print("\nSummary Statistics:\n")
    print(stats)

    print("\nCorrelation Matrix:\n")
    print(corr)

    print("\nCovariance Matrix:\n")
    print(cov)

    print("\nControl Weights:\n")
    print(control_weights)

    print("\nOptimal Weights:\n")
    print(optimal_weights)

    print("\nPortfolio Performance Comparison:\n")
    print(comparison)

    # Plot heatmap
    plot_cum(control_returns, optimized_returns)
    plotcor(corr)


if __name__ == "__main__":
    main()