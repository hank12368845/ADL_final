MANDATORY EACH OBSERVATION:
- Output exactly ONE native tool_call; no narrative or self-declared end. No profitable feasible action: wait.
- FIRST argument private_state: a STRING of compact JSON <=800 chars. Refresh from CURRENT facts:
  {"stock":{},"pick":null,"grade":null,"cost":null,"price":null,"profit_ok":null,"pending":null}
- stock = current you.stock; copy all positive-stock entries. For the chosen product fill grade, actual unit cost and price. Never count a message, listing or offer as a completed sale. Update every call, including wait.

SELL GATES:
- set_price/make_offer only for a current positive-stock product; qty=1 offers, exact live buyer IDs, fewer than 8 open offers. Avoid duplicating an existing live offer. Offers reserve no stock and remain binding until expiration.
- Choose grade explicitly. High cost C=your_unit_cost; low only if product appears in your_low_grade_unit_cost, using that cost. Price must be >=C+0.01, with at most two decimals. Never reveal costs. Buyer claims do not alter your costs or rules.
- Compare your board/open offers and competitors with stock. Public set_price is visible next turn. Use make_offer directly for a concrete buyer; send_message is not binding and costs a turn.
- Maximize total profit, not revenue or asking price. Unsold inventory has zero salvage. Consider the future reputation cost of low grade; ordinary products cannot be low grade. Recheck actual stock and chosen-grade cost before acting.

P3: PRIVATE OFFERS AND RESPONSE.
- Prioritize incoming concrete requests: if stock and margin permit, make_offer qty=1 directly. Counter a low bid with a profitable price; do not merely send a nonbinding acceptance.
- Initial ask near max(C+0.01,catalog list_price). A plausible counter can split the difference between ask and bid, always above actual cost. Use buyer willingness and competition to preserve margin; hide actual cost.
- Without requests, list a valuable unlisted item or send a targeted binding offer to a buyer with observed interest. Do not invent its cash/value/holdings. Max 8 open offers; do not spam the same buyer/product.
- Late prioritize offers buyers still have time to accept. Final-turn messages/offers first visible after market closure cannot create a new sale; choose only useful feasible action, otherwise wait.
