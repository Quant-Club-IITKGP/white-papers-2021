# pip install yfinance
## install yfinance in the local machine if it is not available. To install yfinance run the above code in terminal.

import yfinance as yf
import numpy as np
from datetime import date
import matplotlib.pyplot as plt

#'AAPL','MSFT','GOOG','AMZN','TSLA','FB','NVDA'
## These are some symbols which can be added in the assets array. If you want you can add your own stock tickers in the shown manner.
assets = ['AAPL','MSFT','GOOG','AMZN']

## Block to download the stock data, extract relevant info and make a dataframe out of it
df = yf.download(assets,start="2021-01-31",end=date.today().strftime('%Y-%m-%d'))
stockprice = df['Adj Close']
df = df['Adj Close'].pct_change()

# print(df) 
## if you want to see the df then uncomment the above line while running the code.

## Utility functions
def portfolioreturn(weights):
  return np.dot(df.mean(),weights)

def portfoliostd(weights):
  return np.dot(np.dot(df.cov(),weights),weights)**(1/2)*np.sqrt(252)

def weightcreator(df):
  rand = np.random.random(len(df.columns))
  rand /= rand.sum()
  return rand

def sharperatio(returns,std):
  return returns/std



## Running the simulation for Modern Portfolio Theory Block.
returns = []  ##returns array
stds = []     ##standard deviation array
w = []        ##weights array
r = []        ##Sharpe ratio array
for i in range(500):
  weights = weightcreator(df)
  returns.append(portfolioreturn(weights))
  stds.append(portfoliostd(weights))
  w.append(weights)
  r.append(sharperatio(portfolioreturn(weights),portfoliostd(weights)))


## Output from Modern Portfolio Theory
index = r.index(max(r)) ##index for max sharpe ratio
print(f"Maximaum Sharpe Ration Obtainded: {r[index]}")
print(f"Risk at max Sharpe Ratio: {stds[index]}")
print(f"Returns at max Sharpe Ratio: {returns[index]}")
print(f"Weights of the portfolio by MPT: {w[index]}")


## Plotting Efficient Frontier
plt.figure(figsize=(14,8))
plt.scatter(stds,returns,c=r)
plt.scatter(stds[index],returns[index],c='red')
plt.title("Efficient Frontier")
plt.xlabel("Portfolio risk")
plt.ylabel("Portfolio return")
plt.colorbar(label="SR")
plt.show()


## Application of the weights derived from MPT in real market data
startamt = 10000 #considering start amt to be $10,000 
amtallocation = w[index]*startamt
print(f"Amount allocation in the portfolio: {amtallocation}")

startstockprice = np.array(stockprice.loc['2021-10-01'])
print(f"Stock price on 1 Dec, 2021{startstockprice}")

portfolio = np.divide(amtallocation,startstockprice)
print(f"Your Portfolio: {portfolio}")

todayprice = np.array(stockprice.iloc[-1])
print(f"Stock prices today: {todayprice}")

portfoliovalue=np.sum(np.multiply(portfolio,todayprice))
print(f"Your Portfolio value today: {portfoliovalue}")

marketreturn = portfoliovalue - startamt
print(f"Your returns from the market from the portfolio: ${marketreturn}")

pctreturn = (marketreturn/startamt)*100
print(f"Percentage of your return: {pctreturn} %")