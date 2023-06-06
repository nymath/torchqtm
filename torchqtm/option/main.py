import torchqtm.option.functional as F
from torchqtm.utils import Timer

S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
price = 10.450583572185565  # Assuming the market price of option

# Pricing the option
price = F.bsm_call(T, S, K, r, sigma)
print("The Black-Scholes price is ", price)

# Calculating implied volatility
imp_vol = F.implied_vol(T, S, K, r, price, 'newton')
with Timer():
    print("The implied volatility is ", imp_vol)

# with Timer():
#     df = lambda x: (x-3)**2
#     ddf = lambda x: 2*(x-3)+1
#     tt = F._Newton(df, ddf, x_0=1, N=10000)[-1]
#     print(df(tt))

