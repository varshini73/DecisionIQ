# LinkedIn Project Post (Draft)

---

Most "demand forecasting" projects stop at a prediction. I wanted to build
something that stops at a *decision* instead — because that's the part a
retailer actually has to act on.

So I built **DecisionIQ**: an end-to-end decision intelligence system for
retail inventory.

It doesn't just forecast demand. It:
→ Flags which SKUs are about to stock out, and by when
→ Flags which SKUs are quietly tying up capital in excess inventory
→ Calculates the exact reorder quantity, safety stock, and reorder point
→ Attaches a dollar cost to *not* acting on the recommendation
→ Explains, in plain English, why each recommendation was made

Under the hood: a LightGBM/XGBoost demand model benchmarked against a
seasonal-naive baseline (because if you can't beat "same day last week,"
your model isn't adding value) — feeding into a safety-stock/EOQ
optimization layer — feeding into a rule-based decision engine — surfaced
through a Streamlit dashboard with a live what-if simulator for service
level, lead time, and holding cost trade-offs.

The part I found most interesting: stockout days *censor* demand. If a
product was out of stock, "units sold = 0" doesn't mean "demand = 0" — it
means "we don't actually know." Correcting for that before training the
forecasting model made a real difference, and it's the kind of detail that
doesn't show up in a typical Kaggle notebook.

Full code, architecture, and write-up on GitHub: [link]

Would love to hear from anyone working in decision sciences / inventory
optimization — what's the one real-world business problem you'd tell an
aspiring data scientist to try solving first?

#DecisionScience #DataScience #InventoryOptimization #MachineLearning #Retail

---

**Note:** Fill in your actual FVA/WAPE numbers once you run the pipeline —
a specific number ("beat baseline by 23%") reads stronger than a vague claim.
