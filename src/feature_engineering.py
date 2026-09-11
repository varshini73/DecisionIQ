"""Leakage-safe features and transparent stockout-demand correction."""
import numpy as np
import pandas as pd
from utils import DATA_PROCESSED_DIR, CONFIG, get_logger
logger=get_logger("feature_engineering")

def uncensor_demand(df):
    """Estimate censored demand only where stock was unavailable and sales were zero.
    The estimate is a trailing 14-day mean of *non-stockout* observations;
    it is an assumption, not a recovery of true demand.
    """
    df=df.sort_values(["store_id","sku_id","date"]).copy(); keys=[df.store_id,df.sku_id]
    clean=df.sales.where(df.is_stockout_day.eq(0)); trailing=clean.groupby(keys,observed=True).transform(lambda s:s.shift(1).rolling(14,min_periods=3).mean().ffill())
    df["observed_sales"]=df.sales; df["estimated_demand"]=df.sales
    mask=df.is_stockout_day.eq(1)&trailing.notna(); df.loc[mask,"estimated_demand"]=trailing[mask]
    return df

def add_features(df):
    df=df.sort_values(["store_id","sku_id","date"]).copy(); target="estimated_demand"; group=df.groupby(["store_id","sku_id"],observed=True)[target]
    for n in (1,7,14,28): df[f"lag_{n}"]=group.shift(n)
    shifted=group.shift(1)
    keys=[df.store_id,df.sku_id]
    for n in (7,14,28):
        df[f"rolling_mean_{n}"]=shifted.groupby(keys,observed=True).transform(lambda s:s.rolling(n,min_periods=3).mean())
    for n in (7,28): df[f"rolling_std_{n}"]=shifted.groupby(keys,observed=True).transform(lambda s:s.rolling(n,min_periods=3).std())
    df["day_of_week"]=df.date.dt.dayofweek; df["day_of_month"]=df.date.dt.day; df["week_of_year"]=df.date.dt.isocalendar().week.astype(int); df["month"]=df.date.dt.month; df["quarter"]=df.date.dt.quarter; df["weekend"]=(df.day_of_week>=5).astype(int)
    return df

def add_abc_xyz(df):
    revenue=(df.estimated_demand*df.selling_price).groupby(df.sku_id,observed=True).sum().sort_values(ascending=False); shares=revenue.cumsum()/revenue.sum(); a,b=CONFIG["abc_thresholds"]["A"],CONFIG["abc_thresholds"]["B"]
    abc={sku:("A" if share<=a else "B" if share<=b else "C") for sku,share in shares.items()}; df["abc_class"]=df.sku_id.map(abc)
    stats=df.groupby("sku_id",observed=True).estimated_demand.agg(["mean","std"]); cv=stats["std"]/stats["mean"].replace(0,np.nan); x,y=CONFIG["xyz_thresholds"]["X"],CONFIG["xyz_thresholds"]["Y"]
    df["xyz_class"]=df.sku_id.map({sku:("X" if v<=x else "Y" if v<=y else "Z") for sku,v in cv.fillna(np.inf).items()}); return df

def run_pipeline():
    df=pd.read_parquet(DATA_PROCESSED_DIR/"clean_panel.parquet"); df=add_abc_xyz(add_features(uncensor_demand(df))); df.to_parquet(DATA_PROCESSED_DIR/"features.parquet",index=False); logger.info("Saved %s feature rows",len(df)); return df
if __name__=="__main__": run_pipeline()
