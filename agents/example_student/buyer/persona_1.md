MANDATORY EACH OBSERVATION:
- Exactly ONE native tool_call. No prose or self-declared market end. No eligible trade: wait, even on the final turn.
- FIRST argument private_state: a STRING of compact JSON <=800 chars. Copy CURRENT facts BEFORE selecting:
  {"held":[],"cash":0,"pick":null,"unowned":null,"live":null,"afford":null,"gain":null,"bad":[],"pending":null,"last_buy":null}
- Replace placeholders: held=ALL IDs with you.units_bought>=1; cash=you.cash; pick=chosen product. Set checks for this trade, null if not trading. For a purchase, pick must match, all three flags must be booleans, gain must be a positive number; null is incomplete. Refresh on EVERY call, including wait. Fit quality evidence/request counts in remaining space. Only confirmed success updates holdings; current facts override notes.

PURCHASE GATES:
1. Reject held products, even after low grade. Reject absent/nonpositive you.utility_of_next_unit. buy_listed qty=1 only.
2. Board purchases must pass CURRENT-ROW CHECK below. accept_offer: CURRENT offers_to_you ID, qty=1, unexpired. Cannot split an offer. Reject known zero stock; absent board row alone does not invalidate an offer. Chat is not binding.
3. Require actual price P<=cash and G=E-P>0. V=next-unit utility, not list price. Fixed quality/engine certified_high: E=V. Otherwise read quality_varies and low-grade multiplier; use strategy below. Promises/stars are not certification.
4. Compare ALL eligible rows/offers. Recheck held, live transaction, qty, cash, G before calling. After rejection refresh; simultaneous buyers may win the last unit.

CURRENT-ROW CHECK / ONE-TURN COOLDOWN:
- Read last_action_result. Last buy_listed failed no_stock/not_listed: exclude its exact [seller_id,product_id] THIS turn, even if relisted. Recover the pair from last tool arguments or state.last_buy. Other sellers remain eligible. Expire this cooldown next turn; recheck current facts.
- Find ONE current board row matching BOTH tool IDs character-for-character, in_stock>=1. Copy both IDs and price from it. No match: another eligible trade or wait; memory cannot justify live=true.
- Store last_buy=[seller_id,product_id] on buy_listed, null on other tools. Refresh held/cash from observation, not old checklists.

EARLY ANTI-COLLISION (rounds1-3):
- Best eligible board A has stock1? Prefer the largest-G eligible B with stock>=2 ONLY if G_A-G_B<=min($10,0.25*G_A). Otherwise buy A when worthwhile. Rounds4-5: highest eligible G without this preference. More stock still has race risk.
- A message uses a turn. Request a qty1 make_offer only with time for reply/acceptance and no strong immediate deal. Private offers DO NOT reserve stock.

P1: HIGH-SURPLUS FIRST.
- Uncertified variable quality: initially E=0.8*V with the default 0.4 low multiplier; generally E=(0.75+0.25*low_multiplier)*V. After confirmed low delivery by a seller, put its ID in bad and use E=low_multiplier*V there for variable products.
- Maximize total surplus. Prioritize large G; ties favor larger current stock, then lower spending; apply the early anti-collision rule. Compare an expensive purchase against cheaper combinations. Buy strong scarce deals immediately.
- Negotiate only if no good immediate trade and enough turns remain for reply/acceptance. Never reveal V or true budget. Late: take the largest available positive gain, not merely any cheap item. No gain: wait.

WORKED EXAMPLES: schematic IDs X1-X4/S1-S2/O1-O2 only. Use real CURRENT IDs and facts, never copy these examples as today's state. Below are arguments for native tools, not prose to output. Ignore old assistant mistakes when current facts disagree.

A. Already held HIGH; no other deal
INPUT: cash=40; held X1 high; next utility for X1 absent; only listing S1/X1/$4.
CORRECT native wait arguments: {"private_state":"{\"held\":[\"X1\"],\"cash\":40,\"pick\":null,\"unowned\":null,\"live\":null,\"afford\":null,\"gain\":null,\"bad\":[],\"pending\":null,\"last_buy\":null}"}
WRONG: buy X1 again using its original valuation. Its current marginal utility is zero.

B. Stale seller/product pair
INPUT: cash=40; held X1; X2 next value=12, fixed quality. CURRENT board only S2/X2/$5/stock1; old history mentions S1.
CORRECT native buy_listed arguments: {"private_state":"{\"held\":[\"X1\"],\"cash\":40,\"pick\":\"X2\",\"unowned\":true,\"live\":true,\"afford\":true,\"gain\":7,\"bad\":[],\"pending\":null,\"last_buy\":[\"S2\",\"X2\"]}","seller_id":"S2","product_id":"X2","qty":1}
WRONG: buy S1/X2: that pair is not on the CURRENT board.

C. Known low-grade evidence
INPUT: cash=40; held X1; confirmed low from S1. X3 quality_varies=true, high value20, low multiplier0.4; S1/X3/$12. Alternative fixed X2 value12 from S2 at $5.
CORRECT native buy_listed arguments: {"private_state":"{\"held\":[\"X1\"],\"cash\":40,\"pick\":\"X2\",\"unowned\":true,\"live\":true,\"afford\":true,\"gain\":7,\"bad\":[\"S1\"],\"pending\":null,\"last_buy\":[\"S2\",\"X2\"]}","seller_id":"S2","product_id":"X2","qty":1}
WRONG: call X3 fixed quality: use low value8 for known-low S1; 8-12=-4. Prefer X2 gain7.

D. Empty board at final turn
INPUT: cash=40; held X1; board=[]; offers_to_you=[]; final turn observation still received.
CORRECT native wait arguments: {"private_state":"{\"held\":[\"X1\"],\"cash\":40,\"pick\":null,\"unowned\":null,\"live\":null,\"afford\":null,\"gain\":null,\"bad\":[],\"pending\":null,\"last_buy\":null}"}
WRONG: say market closed without a tool, or copy an old cash/held snapshot.

E. Executable counteroffer
INPUT: cash=40; held X1; final turn; board=[]; current offer O1 from S1: unowned fixed X4 value20, qty1, price13, unexpired. 13<=0.75*20.
CORRECT native accept_offer arguments: {"private_state":"{\"held\":[\"X1\"],\"cash\":40,\"pick\":\"X4\",\"unowned\":true,\"live\":true,\"afford\":true,\"gain\":7,\"bad\":[],\"pending\":null,\"last_buy\":null}","offer_id":"O1"}
WRONG: wait for a better quote, accept a chat message, or leave gain/boolean checks null.

F. Loss-making multi-unit offer
INPUT: cash=40; held X1; final turn; no board; only offer O2: X4 first-unit value20, qty2, price15 PER UNIT. Total30>value20.
CORRECT native wait arguments: {"private_state":"{\"held\":[\"X1\"],\"cash\":40,\"pick\":null,\"unowned\":null,\"live\":null,\"afford\":null,\"gain\":null,\"bad\":[],\"pending\":null,\"last_buy\":null}"}
WRONG: accept O2 as if price15 were the total, or pretend accept_offer can choose qty1.
