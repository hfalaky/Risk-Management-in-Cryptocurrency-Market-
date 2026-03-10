import pandas as pd
from ydata_profiling import ProfileReport 



def loading(filepath: str) -> pd.DataFrame:
   
    df = pd.read_csv(filepath)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.set_index("Date")

    df = df.apply(pd.to_numeric, errors="raise") #excluding date cause index
    return df


def profile(df: pd.DataFrame) -> None:

    report = ProfileReport(df, title="Crypto Dataset Profiling")

    report.to_file("profile_report.html")



# A simple return measures how much an asset’s price changed relative to the previous period.
def ret(prices: pd.DataFrame) -> pd.DataFrame:

    #converts price levels → percentage changes

    returns = prices.pct_change().dropna()
    return returns


prices = loading("closing.csv")
returns = ret(prices)
print(returns.head())