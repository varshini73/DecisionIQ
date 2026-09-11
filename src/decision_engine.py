"""Create auditable operational recommendations from inventory policy."""
import pandas as pd
from utils import DATA_PROCESSED_DIR, get_logger
from business_rules import classify_decision,classify_urgency,classify_status_light,recommended_quantity,priority_label
logger=get_logger("decision_engine")
def run_pipeline():
    df=pd.read_parquet(DATA_PROCESSED_DIR/"inventory_policy.parquet").copy(); df["decision"]=df.apply(lambda r:classify_decision(r.inventory_level,r.reorder_point,r.days_until_stockout,r.lead_time_days),axis=1); df["urgency"]=df.apply(lambda r:classify_urgency(r.decision,r.abc_class),axis=1); df["priority"]=df.urgency.map(priority_label); df["status_light"]=df.days_until_stockout.map(classify_status_light); df["recommended_qty"]=df.apply(lambda r:recommended_quantity(r.decision,r.reorder_point,r.inventory_level,r.eoq),axis=1)
    def reason(r):
        if r.decision=="REORDER_NOW": return f"Inventory ({r.inventory_level:.0f}) covers {r.days_until_stockout:.1f} days, below the {r.lead_time_days:.0f}-day lead time; reorder point is {r.reorder_point:.0f}."
        if r.decision=="REORDER_SOON": return f"Inventory ({r.inventory_level:.0f}) is below the reorder point ({r.reorder_point:.0f})."
        if r.decision=="OVERSTOCKED_REDUCE": return f"Inventory ({r.inventory_level:.0f}) exceeds 2.5 times the reorder point ({r.reorder_point:.0f})."
        if r.decision=="MONITOR": return f"Inventory is above normal operating range; reorder point is {r.reorder_point:.0f}."
        return f"Inventory is within the operating range around reorder point {r.reorder_point:.0f}."
    df["reason"]=df.apply(reason,axis=1); cols=["store_id","sku_id","category","abc_class","xyz_class","decision","urgency","priority","status_light","inventory_level","forecast_demand","reorder_point","safety_stock","recommended_qty","days_until_stockout","lead_time_days","estimated_stockout_cost","annual_holding_cost","reason"]
    out=df[cols].sort_values(["urgency","estimated_stockout_cost"]); out.to_parquet(DATA_PROCESSED_DIR/"decisions.parquet",index=False); out.to_csv(DATA_PROCESSED_DIR/"decisions.csv",index=False); logger.info("Generated %s recommendations",len(out)); return out
if __name__=="__main__": run_pipeline()
