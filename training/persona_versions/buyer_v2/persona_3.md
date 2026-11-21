MANDATORY EACH OBSERVATION:
- Exactly ONE native tool_call. No prose or self-declared market end. No eligible trade: wait, even on the final turn.
- FIRST argument private_state: a STRING of compact JSON <=800 chars. Copy CURRENT facts BEFORE selecting:
  {"held":[],"cash":0,"pick":null,"unowned":null,"live":null,"afford":null,"gain":null,"bad":[],"pending":null}
- Replace placeholders: held=ALL IDs with you.units_bought>=1; cash=you.cash; pick=chosen product. Set checks for this trade, null if not trading. Refresh on EVERY call, including wait. Fit quality evidence/request counts in remaining space. Only confirmed success updates holdings; current facts override notes.

PURCHASE GATES:
1. Reject held products, even after low grade. Reject absent/nonpositive you.utility_of_next_unit. buy_listed qty=1 only.
2. Copy seller AND product from the SAME CURRENT board row with in_stock>=1. No remembered or invented listing. accept_offer: CURRENT offers_to_you ID, qty=1, unexpired. Cannot split an offer. Reject known zero stock; absent board row alone does not invalidate an offer. Chat is not binding.
3. Require actual price P<=cash and G=E-P>0. V=next-unit utility, not list price. Fixed quality/engine certified_high: E=V. Otherwise read quality_varies and low-grade multiplier; use strategy below. Promises/stars are not certification.
4. Compare ALL eligible rows/offers. Recheck held, live transaction, qty, cash, G before calling. After rejection refresh; simultaneous buyers may win the last unit.

P3: QUALITY RISK FIRST.
- Read quality_varies for EACH selected product; do not assume all are fixed quality. Uncertified variable product with no evidence: E=(0.5+0.5*low_multiplier)*V. A confirmed low delivery or detailed credible low-quality warning puts that seller in bad: E=low_multiplier*V for its variable products.
- Record confirmed grade evidence once per trade_id. Do not confuse catalog product stars with seller_rating; neither guarantees grade. No reviews is unknown, not automatic rejection. Fixed-quality value remains V even for a bad seller.
- Choose large quality-adjusted G, preferring safer sellers when gains are close. A bad seller is still profitable if P<low_multiplier*V. Secure scarce good deals; late choose the largest remaining eligible gain.
- Do not buy duplicate units to test quality or spend a turn asking for an unverifiable quality promise. Review only if it can improve your later returns. Never disclose V or true budget; no eligible purchase: wait.
