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

P1: HIGH-SURPLUS FIRST.
- Uncertified variable quality: initially E=0.8*V with the default 0.4 low multiplier; generally E=(0.75+0.25*low_multiplier)*V. After confirmed low delivery by a seller, put its ID in bad and use E=low_multiplier*V there for variable products.
- Maximize total surplus. Prioritize large G; ties favor scarce supply/low stock, then lower spending. Compare an expensive purchase against cheaper combinations. Buy strong scarce deals immediately.
- Negotiate only if no good immediate trade and enough turns remain for reply/acceptance. Never reveal V or true budget. Late: take the largest available positive gain, not merely any cheap item. No gain: wait.
