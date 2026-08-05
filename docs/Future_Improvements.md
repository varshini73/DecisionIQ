# Future Improvements

1. **Multi-echelon inventory optimization** — currently each store-SKU is
   optimized independently; a real network shares stock across a warehouse
   → store hierarchy, which changes the safety-stock math substantially.
2. **Probabilistic forecasting** — replace point forecasts with quantile
   regression (LightGBM quantile objective) to get a full demand
   distribution instead of a single number ± std-dev assumption.
3. **Supplier lead-time variability** — currently a fixed constant; model
   lead time as a distribution too, since late deliveries are often a
   bigger stockout driver than demand volatility itself.
4. **Cross-SKU substitution effects** — if Product A stocks out, some of
   its demand shifts to Product B. Not modeled here; would require a
   basket/substitution model.
5. **Reinforcement learning for dynamic reorder policy** — frame reordering
   as a sequential decision problem (contextual bandit / RL) instead of a
   static formula, so the policy adapts over time.
6. **Real-time API** — wrap the decision engine in a FastAPI service so it
   can be called from a live ERP/POS system instead of run as a batch job.
7. **A/B testing framework** — measure the actual business impact of
   following DecisionIQ's recommendations vs. the retailer's existing
   process, not just offline forecast accuracy.
8. **Automated model retraining & drift monitoring** — trigger retraining
   when WAPE degrades past a threshold, rather than manual reruns.
