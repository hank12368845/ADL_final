"""Read-only full-market loss accounting; never changes scores or estimates causal regret."""
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
from statistics import mean

IDS = [f'example_student_buyer_p{i}' for i in (1, 2, 3)]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cents(value):
    return int(Decimal(str(value)) * 100)


def utility(table, grades, low):
    grades = sorted(grades, key=lambda g: g == 'low')
    return sum(round(table[i] * (low if g == 'low' else 1.))
               for i, g in enumerate(grades) if i < len(table))


def unit_ledger(trades, tables, low):
    """Replay exact per-unit marginal utility, including high-after-low upgrades."""
    held = defaultdict(list)
    ledger = []
    for trade in trades:  # Engine trade order matters within a tick.
        pid = trade['product']
        prior = len(held[pid])
        total_gain = 0
        for index in range(trade['qty']):
            before = utility(tables[pid], held[pid], low)
            repeat = bool(held[pid])
            held[pid].append(trade['grade'])
            gain = utility(tables[pid], held[pid], low) - before
            total_gain += gain
            ledger.append(dict(trade_id=trade['id'], tick=trade['tick'], product=pid, seller=trade['seller'],
                               unit=index + 1, price_cents=trade['price'], utility_cents=gain,
                               net_cents=gain - trade['price'], repeat_unit=repeat,
                               trade_already_owned=prior > 0, grade=trade['grade'], via=trade['via']))
        if total_gain != trade['utility']:
            raise ValueError(f'Engine marginal utility mismatch: {trade["id"]}')
    return ledger


def visible_options(obs, situation, low, cash_limit=None):
    """Positive worst-grade marginal gain visible at observation time, not guaranteed fills."""
    you = obs['you']
    cash = cents(you['cash']) if cash_limit is None else cash_limit
    tables = {p: [cents(v) for v in values] for p, values in situation['your_utility_per_unit'].items()}
    varies = {r['product_id'] for r in situation['market_overview'] if r['quality_varies']}
    options = []
    for channel in ('board', 'offers_to_you'):
        for row in obs.get(channel, []):
            if channel == 'board' and row.get('in_stock', 0) < 1:
                continue
            if channel == 'offers_to_you' and row['open_until_round'] < obs['round']:
                continue
            pid = row['product_id']
            qty = row.get('qty', 1) if channel == 'offers_to_you' else 1
            if not isinstance(qty, int) or qty < 1:
                continue
            price = cents(row['price']) * qty
            if price > cash or pid not in tables:
                continue
            count = you['units_bought'].get(pid, 0)
            grades = you.get('units_by_grade', {}).get(pid)
            if grades is not None:
                previous = [g for g, n in grades.items() for _ in range(n)]
                if len(previous) != count:
                    raise ValueError('Observation holdings/grade mismatch')
            else:
                if count and pid in varies:
                    continue  # Unknown past grades: do not invent a positive lower bound.
                previous = ['high'] * count
            grade = 'high' if pid not in varies or row.get('certified_high') else 'low'
            gain = utility(tables[pid], previous + [grade] * qty, low) - utility(tables[pid], previous, low)
            if gain > price:
                options.append(dict(source=channel, product=pid, seller=row.get('seller'),
                                    offer_id=row.get('offer_id'), qty=qty, total_cents=price,
                                    marginal_lower_bound_cents=gain, gain_lower_bound_cents=gain-price))
    return sorted(options, key=lambda r: -r['gain_lower_bound_cents'])


def turn_stats(rows):
    llm_rows = [r for r in rows if (r.get('llm') or {}).get('outcome') not in (None, 'round_budget_used')]
    attempts = [r for r in rows if r.get('tool') in ('buy_listed', 'accept_offer')]
    cash_errors = sum(r['result'].get('error') == 'insufficient_cash' for r in rows)
    trunc = sum((r.get('llm') or {}).get('stop_reason') == 'length' for r in rows)
    return dict(turns=len(rows), llm_decision_turns=len(llm_rows), purchase_attempts=len(attempts),
                cash_errors=cash_errors, cash_errors_per_turn=cash_errors/len(rows) if rows else None,
                cash_errors_per_purchase_attempt=cash_errors/len(attempts) if attempts else None,
                truncated_turns=trunc, truncation_per_llm_turn=trunc/len(llm_rows) if llm_rows else None,
                length_retry_attempts=sum(len((r.get('llm') or {}).get('length_retries', [])) for r in rows),
                budget_waits=sum((r.get('llm') or {}).get('outcome') == 'round_budget_used' for r in rows),
                no_tool_turns=sum(r.get('tool') is None for r in rows),
                spent_output_tokens=sum((r.get('llm') or {}).get('output_tokens_spent', 0) for r in rows),
                mean_input_tokens=mean((r['llm'].get('usage') or {}).get('input_tokens', 0) for r in llm_rows) if llm_rows else None,
                mean_history_turns=mean(r['llm'].get('history_turns_shown', 0) for r in llm_rows) if llm_rows else None,
                mean_available_round_tokens=mean(r['observation'].get('output_tokens_left_this_round', 0) for r in rows) if rows else None)


def stock_error_kind(tool, args, current_row, trades, tick):
    if tool == 'accept_offer':
        return 'offer_stock_unavailable'
    if current_row is None:
        return 'no_stock_without_current_board_pair'
    try:
        qty = int(args.get('qty', 1))
    except (TypeError, ValueError):
        return 'no_stock_visible_unresolved'
    if qty > current_row['in_stock']:
        return 'requested_quantity_exceeds_visible_stock'
    pair = (args.get('seller_id'), args.get('product_id'))
    won = sum(t['qty'] for t in trades if t['tick']==tick and (t['seller'],t['product'])==pair)
    return 'no_stock_same_tick_exhausted' if won >= current_row['in_stock'] else 'no_stock_visible_unresolved'


def analyze_market(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir/'metadata.json').read_text())
    if meta.get('status') != 'finished':
        raise ValueError('Market run is not finished')
    folders = sorted(p for p in (run_dir/'markets').iterdir() if p.is_dir())
    if len(folders) != 1:
        raise ValueError('This diagnostic requires exactly one market, not extra rotations')
    folder = folders[0]
    final = json.loads((folder/'final.json').read_text())
    events = read_rows(folder/'events.jsonl')
    rows = [r for r in events if r.get('type') == 'turn']
    agents = {r['agent']: r for r in read_rows(folder/'agents.jsonl')}
    scores = {r['agent']: r for r in json.loads((folder/'scores.json').read_text())}
    if len(final['sellers']) != 5 or len(final['buyers']) != 10 or len(rows) != 450:
        raise ValueError('Incomplete 5x10,30-turn market')
    ids = {b['id'] for b in final['buyers']} | {s['id'] for s in final['sellers']}
    if len({(r['agent'], r['tick']) for r in rows}) != 450:
        raise ValueError('Repeated/missing agent turns')
    if {r['agent'] for r in rows} != ids or any(Counter(r['tick'] for r in rows)[t] != 15 for t in range(1, 31)):
        raise ValueError('Incorrect market horizon')
    low = final['schema']['low_utility']
    kinds = {r['product_id']: r['kind'] for r in final['schema']['products']}
    load = defaultdict(Counter)
    for r in rows:
        log = r.get('llm') or {}
        if log.get('base_url'):
            counter = load[log['base_url']]
            counter['calls'] += 1
            counter['spent_output_tokens'] += log.get('output_tokens_spent', 0)
            counter['input_tokens'] += (log.get('usage') or {}).get('input_tokens', 0)
    buyers = {}
    all_scores = {b['id']: scores[b['id']]['raw'] for b in final['buyers']}
    for b in final['buyers']:
        aid = b['id']
        if aid not in IDS:
            continue
        actions = sorted((r for r in rows if r['agent'] == aid), key=lambda r: r['tick'])
        if len(actions) != 30:
            raise ValueError('Persona is missing turns')
        trades = [t for t in final['trades'] if t['buyer'] == aid]
        ledger = unit_ledger(trades, b['utility'], low)
        spent = sum(u['price_cents'] for u in ledger)
        gained = sum(u['utility_cents'] for u in ledger)
        net = gained-spent
        if spent != b['spent'] or gained != b['gained_utility'] or b['cash'] != b['budget']-spent or cents(scores[aid]['raw']) != net:
            raise ValueError(f'Engine surplus/spent/utility/cash mismatch: {aid}')
        zero = [u for u in ledger if u['utility_cents'] == 0]
        useful_net = sum(u['net_cents'] for u in ledger if u['utility_cents'] > 0)
        zero_spend = sum(u['price_cents'] for u in zero)
        assert net == useful_net-zero_spend
        repeat_trades = {u['trade_id'] for u in ledger if u['trade_already_owned']}
        errors = Counter(r['result'].get('error') for r in actions if not r['result'].get('ok'))
        counts = Counter()
        opportunities = []
        details = []
        for r in actions:
            obs = r['observation']; log = r.get('llm') or {}; args = r.get('args') or {}
            tool = r.get('tool'); err = r['result'].get('error'); tick = r['tick']
            current = visible_options(obs, agents[aid]['situation'], low)
            prior_loss = sum(u['price_cents'] for u in zero if u['tick'] < tick)
            shadow = visible_options(obs, agents[aid]['situation'], low, cents(obs['you']['cash'])+prior_loss)
            gap = [o for o in shadow if o['total_cents'] > cents(obs['you']['cash'])]
            turn_trades = [u for u in ledger if u['tick'] == tick]
            pid = args.get('product_id'); seller = args.get('seller_id')
            if tool == 'accept_offer':
                offer = next((o for o in obs.get('offers_to_you', []) if o['offer_id'] == args.get('offer_id')), None)
                if offer:
                    pid, seller = offer['product_id'], offer['seller']
            current_row = next((o for o in obs.get('board', []) if (o['seller'],o['product_id']) == (seller,pid) and o['in_stock'] > 0), None)
            if tool in ('buy_listed', 'accept_offer'):
                counts['already_owned_purchase_attempts'] += obs['you']['units_bought'].get(pid, 0) > 0
            if tool == 'buy_listed':
                counts['absent_board_pair_attempts'] += current_row is None
            if tool == 'accept_offer':
                counts['absent_offer_attempts'] += not any(o['offer_id'] == args.get('offer_id') for o in obs.get('offers_to_you', []))
            if err == 'no_stock':
                counts[stock_error_kind(tool,args,current_row,final['trades'],tick)] += 1
            if err or tool == 'wait' or any(u['utility_cents'] == 0 for u in turn_trades):
                if current:
                    counts['problem_turns_with_visible_positive_lower_bound'] += 1
            if gap:
                counts['turns_with_cash_gap_after_prior_zero_spend'] += 1
                opportunities.append(dict(tick=tick, round=r['round'], cash_cents=cents(obs['you']['cash']),
                    prior_zero_spend_cents=prior_loss, candidates=gap,
                    note='Snapshot affordability sensitivity only. Not causal regret or guaranteed lost surplus; do not sum gains across turns.'))
            details.append(dict(tick=tick, round=r['round'], turn_in_round=(tick-1)%6+1, tool=tool, error=err,
                cash_cents=cents(obs['you']['cash']), selected_product=pid,
                held_before=obs['you']['units_bought'], current_high_next_value=obs['you']['utility_of_next_unit'].get(pid, 0),
                spent_cents=sum(u['price_cents'] for u in turn_trades), utility_cents=sum(u['utility_cents'] for u in turn_trades),
                zero_utility_spend_cents=sum(u['price_cents'] for u in turn_trades if u['utility_cents']==0),
                visible_positive_lower_bound=current, output_tokens_spent=log.get('output_tokens_spent',0),
                stop_reason=log.get('stop_reason'), outcome=log.get('outcome'),
                input_tokens=(log.get('usage') or {}).get('input_tokens'), history_turns_shown=log.get('history_turns_shown'),
                reasoning=log.get('reasoning'), text=log.get('text')))
        per_round = {str(i): turn_stats([r for r in actions if r['round']==i]) for i in range(1,6)}
        for i,s in per_round.items():
            units=[u for u in ledger if (u['tick']-1)//6+1==int(i)]
            s.update(units=len(units),trade_count=len({u['trade_id'] for u in units}),
                     surplus_cents=sum(u['net_cents'] for u in units),
                     zero_marginal_spend_cents=sum(u['price_cents'] for u in units if u['utility_cents']==0))
        if any(s['spent_output_tokens'] > 8192 for s in per_round.values()):
            raise ValueError('Recorded round token spending exceeds official budget')
        per_kind = {}
        for kind in sorted(set(kinds.values())):
            part = [u for u in ledger if kinds.get(u['product']) == kind]
            per_kind[kind] = dict(units=len(part), unique_products=len({u['product'] for u in part}),
                                 first_units=sum(not u['repeat_unit'] for u in part),
                                 surplus_cents=sum(u['net_cents'] for u in part),
                                 zero_marginal_spend_cents=sum(u['price_cents'] for u in part if u['utility_cents']==0))
        buyers[aid] = dict(surplus_cents=net, spent_cents=spent, gained_utility_cents=gained, cash_left_cents=b['cash'],
            rank_among10=1+sum(v>scores[aid]['raw'] for v in all_scores.values()),
            trade_count=len(trades), units=len(ledger), unique_products=len({u['product'] for u in ledger}),
            profitable_unit_net_cents=sum(max(u['net_cents'],0) for u in ledger),
            loss_on_negative_units_cents=sum(max(-u['net_cents'],0) for u in ledger),
            repeat_trade_count=len(repeat_trades), repeat_trade_spend_cents=sum(u['price_cents'] for u in ledger if u['trade_already_owned']),
            repeat_units=len([u for u in ledger if u['repeat_unit']]),
            zero_marginal_units=len(zero), zero_marginal_spend_cents=zero_spend,
            zero_marginal_repeat_spend_cents=sum(u['price_cents'] for u in zero if u['repeat_unit']),
            positive_marginal_repeat_units=sum(u['repeat_unit'] and u['utility_cents']>0 for u in ledger),
            other_units_net_cents=useful_net, counts=dict(counts), errors=dict(errors),
            action_stats=turn_stats(actions), early_turns1_3=turn_stats([r for r in actions if (r['tick']-1)%6<3]),
            late_turns4_6=turn_stats([r for r in actions if (r['tick']-1)%6>=3]),
            by_turn_slot={str(i):turn_stats([r for r in actions if (r['tick']-1)%6+1==i]) for i in range(1,7)},
            by_round=per_round, by_kind=per_kind, cash_gap_snapshots=opportunities,
            unit_ledger=ledger, action_details=details,
            model_failures=scores[aid].get('model_failures',0),
            policy_repair_turns=sum(bool((r.get('llm') or {}).get('parsed_from_text') or (r.get('llm') or {}).get('arguments_repaired')) for r in actions))
    if set(buyers) != set(IDS):
        raise ValueError('Missing target personas')
    return dict(run_dir=str(run_dir.resolve()), market=folder.name, mock=meta.get('mock_llm',False),
                complete=True, actions=450, buyers=buyers, all_buyer_surplus=all_scores,
                actual_server_load={k:dict(v) for k,v in load.items()},
                infrastructure_clean=all(not s.get('model_failures',0) for s in scores.values()))


def write_report(directory, run_dir):
    directory = Path(directory)
    data = analyze_market(run_dir)
    data['valid_real_diagnostic'] = not data['mock'] and data['infrastructure_clean']
    data['combined_target_totals'] = {key:sum(b[key] for b in data['buyers'].values()) for key in (
        'surplus_cents','spent_cents','zero_marginal_spend_cents','zero_marginal_repeat_spend_cents',
        'profitable_unit_net_cents','loss_on_negative_units_cents')}
    data['notes'] = [
        'Engine raw surplus is authoritative. Costs here are already included; never subtract them again.',
        'Already-owned purchase can add utility when high grade replaces an earlier low grade. Zero marginal utility is replayed per unit.',
        'Insufficient-cash rejection and truncation have zero direct transaction spend. Their opportunity cost is not identified by one observed market.',
        'Visible low-grade lower-bound candidates are not guaranteed fills. Cash-gap sensitivity is not causal regret; do not sum across turns.',
        'Turn1-3 vs4-6 is descriptive; history does not reset each round. Actual input tokens/history/token budgets are recorded. Cash depletion and competition confound context effects.',
        'Combined target totals are accounting for these three buyers in this one market, not the course persona score aggregation.',
        'One public market against toy opponents does not establish final grade, a causal aggressive-strategy advantage, or superiority to base.',
        'No automatic next training, deployment or market run.']
    (directory/'market_loss_review.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    money = lambda v: f'{v/100:.2f}'
    lines = ['# Step84完整市場戰損帳單','',
        f'完成：{data["complete"]}；Mock：{data["mock"]}；有效真模型診斷：{data["valid_real_diagnostic"]}。',
        '原persona＋官方step84；一場5seller×10buyer，5round×6turn。','',
        '| Persona | Surplus $ | 名次/10 | 成交 | 支出 $ | 現金 $ | 有利單位淨收益 $ | 負收益單位損失 $ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for aid,b in data['buyers'].items():
        lines.append(f'| {aid} | {money(b["surplus_cents"])} | {b["rank_among10"]} | {b["trade_count"]} | {money(b["spent_cents"])} | '
            f'{money(b["cash_left_cents"])} | {money(b["profitable_unit_net_cents"])} | {money(b["loss_on_negative_units_cents"])} |')
    totals=data['combined_target_totals']
    lines += ['', f'本場三個persona合計surplus：${money(totals["surplus_cents"])}；零邊際效用支出：${money(totals["zero_marginal_spend_cents"])}。這是本場帳目合計，不是課程的persona評分彙整。']
    lines += ['', '## 直接失血與回合損失','',
        '| Persona | 已持有再購成交 | 再購總支出 $ | 零邊際效用支出 $ | 其中重買 $ | 有效升級單位 | 買不起回合 | 截斷 | Token耗盡wait |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for aid,b in data['buyers'].items():
        s=b['action_stats']
        lines.append(f'| {aid} | {b["repeat_trade_count"]} | {money(b["repeat_trade_spend_cents"])} | {money(b["zero_marginal_spend_cents"])} | '
            f'{money(b["zero_marginal_repeat_spend_cents"])} | {b["positive_marginal_repeat_units"]} | {s["cash_errors"]} | {s["truncated_turns"]} | {s["budget_waits"]} |')
    lines += ['', '現金不足／截斷未成交時直接支出為0；回合機會成本不能由這一場精確換算。',
        '帳目恆等式：surplus = 有利單位淨收益 − 負收益單位損失 = 非零效用單位淨收益 − 零效用支出。以上欄位重疊，不可再相加扣分。', '',
        '## 單輪後半段的穩定性','',
        '| Persona | Turn區段 | 回合數 | 購買嘗試 | 買不起 | 模型決策回合 | 截斷 | Token耗盡wait | 平均輸入tokens | 平均實際history |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for aid,b in data['buyers'].items():
        for label,key in [('1–3','early_turns1_3'),('4–6','late_turns4_6')]:
            s=b[key]; inp='n/a' if s['mean_input_tokens'] is None else f'{s["mean_input_tokens"]:.0f}'
            history='n/a' if s['mean_history_turns'] is None else f'{s["mean_history_turns"]:.1f}'
            lines.append(f'| {aid} | {label} | {s["turns"]} | {s["purchase_attempts"]} | {s["cash_errors"]} | {s["llm_decision_turns"]} | '
                         f'{s["truncated_turns"]} | {s["budget_waits"]} | {inp} | {history} |')
    lines += ['', 'JSON另列各turn位置／round的分母、錯誤率、8192輸出預算消耗、實際history與完整問題草稿。',
              'by_kind列稀缺商品等各類成交與surplus，actual_server_load列兩卡實際呼叫與token負載。',
              'cash_gap_snapshots僅列「若保留先前零效用支出，此盤面可能多買得起什麼」，不把它加成確定損失。','',*data['notes']]
    (directory/'market_loss_review.md').write_text('\n'.join(lines)+'\n')
    return data
