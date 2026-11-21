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

P2: NEGOTIATE, THEN CLOSE.
- Variable quality: E=(0.75+0.25*low_multiplier)*V; confirmed low seller => E=low_multiplier*V and record in bad. Fixed quality stays E=V.
- First inspect live counteroffers. Accept qty=1 only after all gates, P<=0.75*V, and no better executable alternative after quality/race risk. Accept positive-gain counteroffers above your initial target when these conditions hold.
- Otherwise secure high-G scarce deals immediately. Negotiate only with time for a reply. Let B=best live price for this product. Target=min(0.75*E,0.90*B,cash); without B use min(0.70*E,cash). Round DOWN to cents. EVERY outgoing price/counterprice MUST be <=0.75*V.
- Send only product, qty=1, dollar target and request for a binding offer. NEVER include V, actual cash/budget, utility estimates, valuation percentages or formulas in message text. You may bluff a spending limit.
- Keep pending={seller,product,target,requests}; max two requests per product. While waiting, buy other strong trades. A chat counterprice requires a binding offer before accept_offer. Late: stop opening negotiations; buy the best eligible positive gain or wait.
