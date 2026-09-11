"""Time-based seasonal-naive vs. gradient-boosted demand forecasting."""
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from utils import DATA_PROCESSED_DIR, MODELS_DIR, CONFIG, get_logger
logger=get_logger("demand_forecasting")
FEATURE_COLS=["lag_1","lag_7","lag_14","lag_28","rolling_mean_7","rolling_mean_14","rolling_mean_28","rolling_std_7","rolling_std_28","day_of_week","day_of_month","week_of_year","month","quarter","weekend","holiday","price","promotion","inventory_level","stock_available","lead_time_days"]
TARGET="estimated_demand"
def wape(y,p): return float(np.abs(np.asarray(y)-np.asarray(p)).sum()/max(np.abs(np.asarray(y)).sum(),1e-9))
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def mae(y,p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
def bias(y,p): return float(np.mean(np.asarray(p)-np.asarray(y)))
def evaluate(y,p,label): return {"model":label,"WAPE":wape(y,p),"RMSE":rmse(y,p),"MAE":mae(y,p)}
def seasonal_naive_forecast(df): return df["lag_7"]
def time_split(df):
    max_date=df.date.max(); test_start=max_date-pd.Timedelta(days=CONFIG["test_days"]); val_start=test_start-pd.Timedelta(days=CONFIG["validation_days"])
    return df[df.date<val_start],df[(df.date>=val_start)&(df.date<test_start)],df[df.date>=test_start]
def train_gbm(train,val):
    """Use LightGBM if installed; sklearn's reliable histogram GBM otherwise."""
    try:
        import lightgbm as lgb
        model=lgb.LGBMRegressor(n_estimators=300,learning_rate=.04,num_leaves=31,random_state=CONFIG["random_seed"],verbosity=-1); model.fit(train[FEATURE_COLS],train[TARGET],eval_set=[(val[FEATURE_COLS],val[TARGET])],callbacks=[lgb.early_stopping(25,verbose=False)]); return model,"LightGBM"
    except ImportError:
        model=HistGradientBoostingRegressor(max_iter=250,learning_rate=.06,max_leaf_nodes=31,l2_regularization=.2,random_state=CONFIG["random_seed"]); model.fit(train[FEATURE_COLS],train[TARGET]); return model,"Histogram Gradient Boosting (fallback)"
def run_pipeline():
    df=pd.read_parquet(DATA_PROCESSED_DIR/"features.parquet").dropna(subset=FEATURE_COLS+[TARGET]); train,val,test=time_split(df)
    if min(len(train),len(val),len(test))==0: raise ValueError("Insufficient history for configured time split")
    baseline=seasonal_naive_forecast(test); mask=baseline.notna(); model,name=train_gbm(train,val); pred=np.clip(model.predict(test[FEATURE_COLS]),0,None)
    results=[evaluate(test.loc[mask,TARGET],baseline[mask],"Seasonal Naive"),evaluate(test[TARGET],pred,name)]; results[1]["FVA_vs_baseline"]=(results[0]["WAPE"]-results[1]["WAPE"])/results[0]["WAPE"]
    comparison=pd.DataFrame(results); comparison.to_csv(DATA_PROCESSED_DIR/"model_comparison.csv",index=False); joblib.dump(model,MODELS_DIR/"demand_model.joblib")
    out=test[["date","store_id","sku_id","category","abc_class","xyz_class","inventory_level","lead_time_days","unit_cost","selling_price",TARGET]].copy(); out["actual_demand"]=out[TARGET]; out["forecast_demand"]=pred; out["baseline_forecast"]=baseline.where(baseline.notna(), pd.Series(pred,index=test.index)); out["model_name"]=name; out.to_parquet(DATA_PROCESSED_DIR/"forecast_results.parquet",index=False)
    logger.info("%s",comparison.to_string(index=False)); return comparison,out
if __name__=="__main__": run_pipeline()
