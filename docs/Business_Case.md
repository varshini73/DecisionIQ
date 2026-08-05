# Business Case

## The problem, in business terms

A mid-size retailer running thousands of SKU-store combinations makes
inventory decisions largely on gut feel or simple moving-average reorder
rules. This produces two costly failure modes simultaneously, on different
SKUs, every week:

- **Stockouts** on fast-moving items → lost sales, damaged customer trust,
  and in competitive categories, permanent share loss to a competitor.
- **Overstock** on slow-moving or over-forecasted items → cash tied up in
  inventory, higher holding costs, and markdown/write-off risk as goods age.

Both failure modes are usually invisible until the P&L review — by which
point the decision that caused them was made weeks earlier.

## What DecisionIQ changes

Instead of a dashboard someone checks occasionally, DecisionIQ produces a
**standing list of decisions**, ranked by urgency and dollar impact, that a
category manager or supply chain lead can act on directly:

- Every SKU-store combination gets one of five clear calls: `REORDER_NOW`,
  `REORDER_SOON`, `HOLD`, `MONITOR`, or `OVERSTOCKED_REDUCE`.
- Every call is backed by a specific number: days of supply remaining vs.
  supplier lead time, and an estimated dollar cost if the recommendation is
  ignored.
- A-class SKUs (the ~20% driving ~80% of revenue) are automatically
  prioritized over C-class SKUs with the same raw numbers — because a
  stockout on a top revenue driver is not equivalent to a stockout on a
  slow mover.

## How this would be piloted with a real retailer

1. **Scope**: start with one category (e.g., a single department across
   10-20 stores) rather than the full catalog, to validate the approach
   before scaling.
2. **Baseline**: measure the retailer's current stockout rate, overstock
   rate, and inventory turnover for that category over the prior quarter.
3. **Run in shadow mode**: generate DecisionIQ recommendations alongside
   the existing process for 4-6 weeks without acting on them, to measure
   forecast accuracy (WAPE, FVA vs. their current method) on live data.
4. **A/B rollout**: apply DecisionIQ recommendations to half the stores in
   the pilot category, keep the other half on the existing process, and
   compare stockout rate, overstock rate, and holding cost after 8-12 weeks.
5. **Quantify impact**: translate the A/B delta into $ — this is the number
   that justifies (or doesn't) a wider rollout.

## Assumptions this project makes (and a real engagement would validate)

- Supplier lead time is treated as roughly constant per SKU — a real
  engagement would pull actual lead-time variability from PO history.
- Unit cost and margin are placeholders where the raw dataset doesn't
  include them — a real engagement would use the retailer's actual cost
  data, which materially changes EOQ and holding-cost calculations.
- Demand is treated as independent across SKUs — no substitution effects
  are modeled (see `docs/Future_Improvements.md`).

Stating these assumptions explicitly, rather than hiding them in code, is
itself part of what makes this feel like a consulting deliverable rather
than a black-box model dump.
