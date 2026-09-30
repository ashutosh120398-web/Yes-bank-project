import os; os.makedirs("figures", exist_ok=True)
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, mplfinance as mpf
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
sns.set_theme(style="whitegrid")
out = {}
data = pd.read_csv("data_YesBank_StockPrices.csv")
out['shape'] = data.shape
out['dups'] = int(data.duplicated().sum())
out['nulls'] = data.isnull().sum().to_dict()
out['dtypes'] = data.dtypes.astype(str).to_dict()
out['describe'] = data.describe().round(3).to_dict()
data['Date'] = pd.to_datetime(data['Date'], format='%b-%y')
df = data.copy().sort_values('Date').set_index('Date')
out['period'] = [str(df.index.min().date()), str(df.index.max().date())]
def save(n): plt.tight_layout(); plt.savefig(f'figures/{n}.png', dpi=150); plt.close()
plt.figure(figsize=(12,5)); plt.plot(df.index, df['Close'], color='black')
plt.title('Yes Bank Closing Price Over Time'); plt.xlabel('Date'); plt.ylabel('Closing Price'); save('fig1_close')
plt.figure(figsize=(12,6))
for c,col in [('Open','blue'),('High','green'),('Low','red'),('Close','black')]: plt.plot(df.index, df[c], label=c, color=col)
plt.title('Open, High, Low and Close Over Time'); plt.legend(); save('fig2_ohlc')
corr = df[['Open','High','Low','Close']].corr(); out['corr']=corr.round(4).to_dict()
plt.figure(figsize=(6,5)); sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f'); plt.title('Correlation Heatmap'); save('fig3_corr')
fig, axes = plt.subplots(2,2, figsize=(12,7))
for ax,col in zip(axes.ravel(), ['Open','High','Low','Close']): sns.boxplot(x=df[col], ax=ax); ax.set_title(f'Box Plot of {col}')
save('fig4_box')
mpf.plot(df[['Open','High','Low','Close']], type='candle', style='charles', ylabel='Price', title='Yes Bank Monthly Candlestick Chart', figsize=(12,6), savefig=dict(fname='figures/fig5_candle.png', dpi=150))
plt.figure(figsize=(12,6)); plt.plot(df.index, df['Close'], label='Close', color='lightgray')
for w,c in [(3,'tab:blue'),(6,'tab:orange'),(12,'tab:red')]: plt.plot(df.index, df['Close'].rolling(w).mean(), label=f'{w}-month MA', color=c)
plt.title('Closing Price with 3, 6 and 12-Month Moving Averages'); plt.legend(); save('fig6_ma')
df['Return'] = df['Close'].pct_change()*100
fig, axes = plt.subplots(1,2, figsize=(14,4)); axes[0].plot(df.index, df['Return'], color='purple'); axes[0].set_title('Monthly Return (%)')
sns.histplot(df['Return'].dropna(), bins=30, kde=True, ax=axes[1]); axes[1].set_title('Distribution of Monthly Returns'); save('fig7_returns')
r = df['Return'].dropna()
out['ret'] = dict(mean=r.mean(), std=r.std(), min=r.min(), max=r.max(), skew=float(stats.skew(r)), kurt=float(stats.kurtosis(r)), min_date=str(r.idxmin().date()), max_date=str(r.idxmax().date()), pos=int((r>0).sum()), neg=int((r<0).sum()))
out['close'] = dict(max=df['Close'].max(), max_date=str(df['Close'].idxmax().date()), min=df['Close'].min(), min_date=str(df['Close'].idxmin().date()), first=df['Close'].iloc[0], last=df['Close'].iloc[-1])
alpha=0.05
rr,p1 = stats.pearsonr(df['Close'].iloc[:-1], df['Close'].shift(-1).iloc[:-1]); out['t1']=[rr,p1]
before = df.loc[:'2017-12-31','Return'].dropna(); after = df.loc['2018-01-01':,'Return'].dropna()
s,p2 = stats.levene(before, after); out['t2']=[before.std(), after.std(), p2, len(before), len(after)]
t,p3 = stats.ttest_ind(before, after, equal_var=False); out['t3']=[before.mean(), after.mean(), p3, t]
m = df[['Open','High','Low','Close']].copy()
for l in (1,2,3): m[f'Close_lag{l}'] = m['Close'].shift(l)
m['MA3'] = m['Close'].rolling(3).mean(); m['Target'] = m['Close'].shift(-1); m = m.dropna()
X = m.drop(columns='Target'); y = m['Target']; sp = int(len(m)*0.8)
Xtr,Xte,ytr,yte = X.iloc[:sp],X.iloc[sp:],y.iloc[:sp],y.iloc[sp:]
out['split']=dict(n=len(m), ntr=len(Xtr), nte=len(Xte), tr0=str(Xtr.index.min().date()), tr1=str(Xtr.index.max().date()), te0=str(Xte.index.min().date()), te1=str(Xte.index.max().date()), features=list(X.columns))
def ev(name, yt, yp):
    return {'Model':name,'RMSE':np.sqrt(mean_squared_error(yt,yp)),'MAE':mean_absolute_error(yt,yp),'R2':r2_score(yt,yp),'Direction accuracy':np.mean(np.sign(np.asarray(yp)-Xte['Close'])==np.sign(yt-Xte['Close']))}
preds = {'Baseline (last close)': Xte['Close'].values}
lin = LinearRegression().fit(Xtr,ytr); preds['Linear Regression']=lin.predict(Xte)
rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(Xtr,ytr); preds['Random Forest']=rf.predict(Xte)
res = pd.DataFrame([ev(n,yte,p) for n,p in preds.items()]).sort_values('RMSE').reset_index(drop=True)
out['results']=res.round(4).to_dict('records')
out['lin_coef']=dict(zip(X.columns, lin.coef_.round(4))); out['lin_int']=float(lin.intercept_)
out['rf_imp']=dict(zip(X.columns, rf.feature_importances_.round(4)))
plt.figure(figsize=(12,6)); plt.plot(yte.index, yte, label='Actual', color='black', linewidth=2)
for n,p in preds.items(): plt.plot(yte.index, p, label=n, linestyle='--')
plt.title('Actual vs Predicted Next-Month Closing Price (Test Period)'); plt.legend(); save('fig8_pred')
fig,ax=plt.subplots(figsize=(8,4)); ax.barh(list(X.columns), rf.feature_importances_); ax.set_title('Random Forest Feature Importance'); save('fig9_imp')
plt.figure(figsize=(12,4)); plt.plot(yte.index, yte-preds['Linear Regression'], color='tab:red'); plt.axhline(0,color='k'); plt.title('Linear Regression Residuals (Test Period)'); save('fig10_resid')
best = 'Linear Regression' if res[res.Model=='Linear Regression'].RMSE.iloc[0] <= res[res.Model=='Random Forest'].RMSE.iloc[0] else 'Random Forest'
out['best']=best
fm = (LinearRegression() if best=='Linear Regression' else RandomForestRegressor(n_estimators=300, random_state=42)).fit(X,y)
last = X.iloc[[-1]]; out['last_pred']=float(fm.predict(last)[0]); out['last_row']=last.round(3).to_dict('records')[0]
# extras: in-sample linear R2 and train perf
out['lin_train_r2']=float(lin.score(Xtr,ytr)); out['lin_test_r2']=float(lin.score(Xte,yte))
json.dump(out, open('results.json','w'), indent=1, default=lambda o: float(o) if isinstance(o,(np.floating,np.integer)) else str(o))
print(json.dumps(out, indent=1, default=lambda o: float(o) if isinstance(o,(np.floating,np.integer)) else str(o)))
