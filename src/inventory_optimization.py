"""Inventory policy formulas, all driven by configurable assumptions."""
import numpy as np
import pandas as pd
from utils import DATA_PROCESSED_DIR, CONFIG, get_logger
logger=get_logger("inventory_optimization")
def compute_safety_stock(df, lead_time_map=None):
    df=df.copy(); df["lead_time_days"]=df.apply(lambda r: lead_time_map.get(r.sku_id, r.lead_time_days) if lead_time_map else r.lead_time_days,axis=1); df["safety_stock"]=(CONFIG["service_level_z"]*df.forecast_error_std*np.sqrt(df.lead_time_days)).clip(lower=0).round(); return df
def compute_reorder_point(df): df=df.copy(); df["reorder_point"]=(df.avg_daily_demand*df.lead_time_days+df.safety_stock).round(); return df
def compute_eoq(df, unit_cost_col="unit_cost"):
    df=df.copy(); cost=df[unit_cost_col] if unit_cost_col in df else 10.; holding=CONFIG["holding_cost_rate"]*cost; denom=holding.replace(0,np.nan) if hasattr(holding,"replace") else holding; df["eoq"]=np.sqrt(2*(df.avg_daily_demand*365)*CONFIG["ordering_cost_per_order"]/denom).fillna(1).round().clip(lower=1); return df
def run_pipeline():
    hist=pd.read_parquet(DATA_PROCESSED_DIR/"forecast_results.parquet").copy(); hist["forecast_error"]=hist.actual_demand-hist.forecast_demand
    snapshot=hist.sort_values("date").groupby(["store_id","sku_id"],observed=True).tail(1).copy(); stats=hist.groupby(["store_id","sku_id"],observed=True).agg(avg_daily_demand=("forecast_demand","mean"),forecast_error_std=("forecast_error","std")).reset_index(); snapshot=snapshot.merge(stats,on=["store_id","sku_id"]); snapshot.forecast_error_std=snapshot.forecast_error_std.fillna(hist.forecast_error.std()).fillna(0)
    snapshot=compute_eoq(compute_reorder_point(compute_safety_stock(snapshot))); snapshot["days_until_stockout"]=(snapshot.inventory_level/snapshot.avg_daily_demand.replace(0,np.nan)).fillna(np.inf); snapshot["annual_holding_cost"]=snapshot.safety_stock*CONFIG["holding_cost_rate"]*snapshot.unit_cost; shortfall=(snapshot.avg_daily_demand*snapshot.lead_time_days-snapshot.inventory_level).clip(lower=0); snapshot["estimated_stockout_cost"]=shortfall*snapshot.selling_price*CONFIG["stockout_margin_loss_rate"]
    snapshot.to_parquet(DATA_PROCESSED_DIR/"inventory_policy.parquet",index=False); logger.info("Saved policy for %s store-SKU combinations",len(snapshot)); return snapshot
if __name__=="__main__": run_pipeline()
