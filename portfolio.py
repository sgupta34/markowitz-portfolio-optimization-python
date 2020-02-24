
import pandas as pd
import numpy as np
import quandl
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.optimize import minimize
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Dates for which stock data is collected
start = pd.to_datetime('2013-01-01')
end = pd.to_datetime('2019-01-01')
# Collecting stock data using QUANDL
aapl = quandl.get('WIKI/AAPL.11',start_date=start,end_date=end)
nike = quandl.get('WIKI/NKE.11',start_date=start,end_date=end)
intel = quandl.get('WIKI/INTC.11',start_date=start,end_date=end)
visa = quandl.get('WIKI/V.11',start_date=start,end_date=end)
msft = quandl.get('WIKI/MSFT.11',start_date=start,end_date=end)
hodp = quandl.get('WIKI/HD.11',start_date=start,end_date=end)
disc = quandl.get('WIKI/DIS.11',start_date=start,end_date=end)
ba = quandl.get('WIKI/BA.11',start_date=start,end_date=end)
pfizer = quandl.get('WIKI/PFE.11',start_date=start,end_date=end)
jnj = quandl.get('WIKI/JNJ.11',start_date=start,end_date=end)
# Merging all the stock to for one file.
stock = pd.concat([aapl,nike,intel,visa,msft,hodp,disc,ba,pfizer,jnj],axis=1)
stock.columns = ['Apple','Nike', 'Intel', 'Visa','Microsoft','Home Depot','Walt Disney','Boeing Company', 'Pfizer Inc','J&J']
# DIsplay the head of the data frame. 
#display(stock.head())
print(stock.head())
#graph of 10 Stocks
f=plt.figure(figsize=(50,30))
plt.plot(stock,linewidth=5)
plt.title('Stock Prices over the years',fontsize=50)
plt.xticks(fontsize=18,rotation=60)
plt.yticks(fontsize=24)
plt.ylabel('Price in USD',fontsize=50)
plt.legend(stock.columns ,loc=2, prop={'size': 30})
print('\nStock Prices over the years Generated...............-\n')
# reating daily return.
print("----------------------------Daily return--------------------")
log_ret =np.log(stock/stock.shift(1))
print(log_ret.head())

log_ret.hist(bins=100,figsize=(12,8))
g = plt.tight_layout()
print('\nDaily Return graph generated.............\n')
print('--------------------Description of Log Return-------------')
print(log_ret.describe().transpose())
print("\n-------------Yearly covariance-----------")
# Compute pairwise covariance of columns
print(log_ret.cov()*252)
# predicting charp ratio using random values of weight and scaling it to 1
np.random.seed(1276)
# Finding optimum in 25000 repititions
num_ports = 25000
all_weight = np.zeros((num_ports,len(stock.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharp_arr = np.zeros(num_ports)
for i in range(num_ports):
    weight = np.array(np.random.random(10))
    weight = weight/np.sum(weight)
    #Save the weight
    all_weight[i,:]=weight
    # Expected Return
    ret_arr[i] = np.sum( (log_ret.mean()* weight)*252)
    #Expected Volitility
    vol_arr[i] = np.sqrt(np.dot(weight,np.dot(log_ret.cov()*252,weight)))
    #Sharp Ratio
    sharp_arr[i]= ret_arr[i]/vol_arr[i]

print("\n====================Random generated=================================\n")
print("Maximum Sharp Ratio(using random number generation) :",sharp_arr.max())
print("Maximum Sharpe Ratio Portfolio Allocation\n")
print('Portfolio allocation graph generate...')
max_sharpe_allocation = pd.DataFrame(all_weight[sharp_arr.argmax()],index=stock.columns,columns=['allocation'])
max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
print(max_sharpe_allocation)
max_sr_ret = ret_arr[sharp_arr.argmax()]
max_sr_vol = vol_arr[sharp_arr.argmax()]
l=plt.figure(figsize=(20,10))
plt.scatter(vol_arr,ret_arr,c=sharp_arr,cmap='plasma')
plt.colorbar(label='Sharp Ratio')
plt.xlabel('Volitility')
plt.ylabel('Return')
plt.grid(True)
plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')
print('Optimization using random ssampling graph generate...')
print("\n=========================================================================\n")
# Generate pie charf for east understanding of portfolio
df = DataFrame (max_sharpe_allocation)
p = plt.figure(figsize=(10,10))
plt.pie(df['allocation'],labels=df.index,autopct='%1.2f%%')
plt.legend()
plt.title("Portfolio Allocation")

#Mathematical Optimization

def get_ret_vol_sr(weight):
    weight = np.array(weight)
    ret = np.sum(log_ret.mean()*weight) *252
    vol = np.sqrt(np.dot(weight.T,np.dot(log_ret.cov()*252,weight)))
    sr=ret/vol
    return np.array([ret,vol,sr])

def neg_sharp(weight):
    return get_ret_vol_sr(weight)[2]*-1


def check_sum(weight):
    # if sum is one it returns zero
    return np.sum(weight)-1

cons = ({'type':'eq','fun':check_sum})
bound = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
init_guess = [0.10,0.10,0.10,0.10,.10,0.10,0.10,0.10,0.10,.10]
opt_result = minimize(neg_sharp,init_guess,method='SLSQP',bounds=bound,constraints=cons)

print("\n-------Mathematical Optimized Result set---------\n")
print(opt_result)
print("\n========================Mathematically Maximized===============================\n")
print("Mathematically Maximized Sharp ratio : ",list(get_ret_vol_sr(opt_result.x))[2])
print("Maximum Sharpe Ratio Portfolio Allocation\n")
max_sharpe_allocation_new = pd.DataFrame(list(opt_result.x),index=stock.columns,columns=['allocation'])
max_sharpe_allocation_new.allocation = [round(i*100,2)for i in max_sharpe_allocation_new.allocation]
print(max_sharpe_allocation_new)
print('Portfolio allocation graph generate...')
df = DataFrame (max_sharpe_allocation_new)
o = plt.figure(figsize=(10,10))
plt.pie(df['allocation'],labels=df.index,autopct='%1.2f%%')
plt.legend()
plt.title("Portfolio Allocation")
#Efficient forointier

frointier_y = np.linspace(.10,.25,50)


def minimize_vol(weight):
    return get_ret_vol_sr(weight)[1]

frointier_vol = []
for possible_return in frointier_y:
    cons = ({'type':'eq','fun':check_sum},
       {'type':'eq','fun': lambda w: get_ret_vol_sr(w)[0]-possible_return})
    result = minimize(minimize_vol,init_guess,method='SLSQP',bounds=bound,constraints=cons)
    frointier_vol.append(result['fun'])
    
p=plt.figure(figsize=(20,10))
plt.scatter(vol_arr,ret_arr,c=sharp_arr,cmap='plasma')
plt.colorbar(label='Sharp Ratio')
plt.xlabel('Volitility')
plt.ylabel('Return')
plt.grid(True)
plt.plot(frointier_vol,frointier_y,marker='^')
print("\nEfficient forointier graph generated... ")
print("\n=========================================================================\n")
plt.show()

input()

