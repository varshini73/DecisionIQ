"""Validate and clean the canonical DecisionIQ retail panel."""
import json
import pandas as pd
from utils import DATA_RAW_DIR, DATA_PROCESSED_DIR, get_logger

logger = get_logger("data_preprocessing")
REQUIRED = {"date","store_id","sku_id","category","sales","price","promotion","stock_available","inventory_level","lead_time_days","unit_cost","selling_price","holiday"}

def validate_schema(df):
    missing = REQUIRED - set(df.columns)
    if missing: raise ValueError(f"Raw data missing required columns: {sorted(missing)}")
    df = df.copy(); df["date"] = pd.to_datetime(df["date"], errors="raise")
    for c in REQUIRED - {"date","store_id","sku_id","category"}: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def reindex_calendar(df):
    frames=[]
    for (store,sku), g in df.groupby(["store_id","sku_id"], observed=True):
        g=g.sort_values("date").set_index("date")
        g=g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D")); g.index.name="date"; g["store_id"],g["sku_id"]=store,sku
        g["sales"]=g["sales"].fillna(0); g["stock_available"]=g["stock_available"].fillna(1)
        for c in ["promotion","holiday"]: g[c]=g[c].fillna(0)
        for c in ["category","price","inventory_level","lead_time_days","unit_cost","selling_price"]: g[c]=g[c].ffill().bfill()
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True)

def cap_outliers_per_series(df, column="sales"):
    grouped=df.groupby(["store_id","sku_id"],observed=True)[column]
    q1=grouped.transform(lambda s:s.quantile(.25)); q3=grouped.transform(lambda s:s.quantile(.75)); median=grouped.transform("median")
    df=df.copy(); df[column]=df[column].clip(upper=(q3+3*(q3-q1)).where((q3+3*(q3-q1))>median+1,median+1)); return df

def validation_statistics(df, duplicates):
    return {"stores":int(df.store_id.nunique()),"skus":int(df.sku_id.nunique()),"date_min":str(df.date.min().date()),"date_max":str(df.date.max().date()),"duplicates_removed":int(duplicates),"missing_values":int(df.isna().sum().sum()),"zero_sales_pct":round(float((df.sales==0).mean()*100),2),"stockout_pct":round(float(((df.stock_available==0)&(df.sales==0)).mean()*100),2)}

def run_pipeline(raw_filename="retail_inventory_raw.csv"):
    df=validate_schema(pd.read_csv(DATA_RAW_DIR/raw_filename)); before=len(df); df=df.drop_duplicates(["store_id","sku_id","date"]); df=reindex_calendar(df); df=cap_outliers_per_series(df); df["is_stockout_day"]=((df.stock_available<=0)&(df.sales<=0)).astype(int); df=df.sort_values(["store_id","sku_id","date"]).reset_index(drop=True)
    stats=validation_statistics(df,before-len(df)); df.to_parquet(DATA_PROCESSED_DIR/"clean_panel.parquet",index=False); (DATA_PROCESSED_DIR/"validation_stats.json").write_text(json.dumps(stats,indent=2)); logger.info("Saved clean panel (%s rows): %s",len(df),stats); return df

if __name__ == "__main__": run_pipeline()
