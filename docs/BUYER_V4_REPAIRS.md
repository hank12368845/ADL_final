# Buyer v4：掛牌、記憶、競購與雙卡（2026-11-18）

## 已確認的本場問題

來源：`results/basic_checks/20261118-162537_gpu_buyer_e1iel219/run_market_20261118-162538`。
逐筆結果在該 run 的 `buyer_failure_audit.json`，原始紀錄仍保留。

| Buyer | no_stock：同 tick 成交耗盡觀測庫存 | no_stock：當前無此掛牌 | not_listed | 本次明確輸出 state | 向 extra 買入／其中 low |
|---|---:|---:|---:|---:|---:|
| p1 | 10 | 3 | 4 | 30/30 | 4/0 |
| p2 | 8 | 2 | 4 | 1/30 | 5/0 |
| p3 | 8 | 8 | 0 | 30/30 | 5/0 |

p1 第28～30 tick 連續買 `ta_seller_p2/P47`，board 已無此組合。
另10次 no_stock 均找到同 tick 的其他成交，數量足以耗盡觀測庫存；不是沒看庫存的同一類錯誤。
p3 也有連續重試失敗 pair，不能只修 p1。
p2 第2～30 tick 沒有明確提供新的 private_state；event 中仍留著第一回合的 state，不能把沿用舊值算成更新成功。

向 extra 的14筆全部 high，surplus 分別為22.73／49.18／78.68。
其中只有3筆是品質可變商品，其餘固定品質；不能直接把14/14當成品質可變商品的高品質機率估計。
extra_2 全市場另售出兩筆 low 給其他買家，所以不能將 extra 身分視為品質保證。
學生的兩筆 low 均來自 ta_seller_p3：p1、p3 各買 P65，價格4.50、增量效用16.40，各賺11.90。
本場無學生因 low 交易產生負 surplus。

## Persona 改動

V3 原文已備份至 `training/persona_versions/buyer_v3/`；active 現為 v4。

- 三份皆加入 last_action_result 的下一回合冷卻：上一筆 buy_listed 的 no_stock／not_listed pair 本回合跳過，其他 seller 仍可買。只冷卻一回合，不建立永久封鎖。`last_buy` 保存實際 seller/product，避免只有商品 ID 無法辨識失敗來源。
- buy_listed 的兩個 ID 必須逐字來自同一個當前有貨 row；記憶不能支撐 live=true。
- Round1～3 最高 G 的 row 庫存1時，如果有庫存至少2且收益少不超過 `min($10, 0.25*G_best)` 的合法替代，優先替代；Round4～5取消此偏好。相較固定容忍少5～10元，比例上限避免低收益交易被換成幾乎沒收益的交易。
- 私訊只在無強立即交易且來得及回覆／接受時使用。make_offer **不保留庫存**；accept_offer 仍呼叫當下的 `_trade`，不能保證免競購。
- p2 強調每次工具都重新輸出 state，並修正既有訊息示例中 pending target=10、實際報價=15的不一致。
- p3 新賣家高品質機率 q=2/3，`E=(q+(1-q)*low_multiplier)*V`；預設低品質倍率0.4時，初始 E 從0.7V改為0.8V。有確認 low 仍降至0.4V。h/l 僅從 agent 自己可見的實際交付累計，不能利用 final.json 的隱藏品質。

此先驗是策略假設，尚無實測品質機率保證。p3 已向 extra 賺78.68，占自身120.50約65%；
不能由單場分數證明是先驗過重導致低分。缺貨避讓也可能錯過本來搶得到的好交易，需比較後保留。
這些是模型應遵從的 persona 指令，沒有改寫 engine 的工具合法性，也未在模型外修復動作。

## 雙卡修正

原分流是學生3人一張、TA12人一張：生成量10471／157146 tokens，約6.2%／93.8%。
新版 `training/market_load_balance.py` 保留學生 persona 的原始展開，按各 TA agent 的平均生成量分配。
`--load-profile` 可指定完成的真實 run；未提供時使用角色／thinking 的初始負載估計，拒絕用 mock 分數作 profile。
產生 session 內的 TA YAML 與兩卡各自的 adapter 路徑，避免原 loader 將相同 LoRA 路徑去重後全送回同卡。
權重使用 hardlink；跨檔案系統改為完整複製並核對 SHA256。沒有修改原 TA 權重。
驗證前後 agent ID／順序／owner／persona／thinking／解碼／市場規則一致，實際 loader 路由必須符合分配。

以本場為 profile 的 plan-only：GPU0 10 agents、GPU1 5 agents；預計每 tick 生成量2766.4／2820.9，約49.5%／50.5%。
這是總體靜態估計，沒有涵蓋每 tick 波動、prefill、不同 GPU 算力或下一場行為改變；實際加速尚未測量。
`resource_plan.json` 記錄預估及路由，結束後 `buyer_failure_audit.json` 記錄實際每 endpoint 的 calls／tokens。

## 驗證與下一次執行

已完成：3個 CPU 路由回歸測試；buyer／seller／品質測試配置路由核對；正式5×10設定的450 actions mock；
實機唯讀 plan-only；六個失敗／競購對照案例的 v3/v4 配對輸入準備。這些不代表 v4 收益改善。

先以同一批凍結輸入比較 v3、v4，144 calls，預設雙卡、有進度與 ETA；由使用者執行：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_diagnostics.py \
  --versions v3,v4 \
  --cases-file results/basic_checks/20261118-162537_gpu_buyer_e1iel219/run_market_20261118-162538/failure_replay_cases.json
```

六題含三人的錯配、p2缺 state／買已持有品、p1合法掛牌競購對照。觀測與原歷史／去歷史各測一次，兩個seed。
診斷的正確動作不要求預知同 tick 其他人的行動。固定題不能測得整場避碰收益或長期記憶遵從；
既有診斷 rubric 也不會把所有策略偏好當成引擎違規。
模型輸出可解析且檢查改善後，再跑一場原規模完整30turns市場確認實際surplus：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/basic_checks.py run \
  --mode gpu --role buyer --config configs/buyer_scored_short.yaml \
  --load-profile results/basic_checks/20261118-162537_gpu_buyer_e1iel219/run_market_20261118-162538
```

兩指令依序執行，不同時搶相同 port。輸出／log 路徑皆由 wrapper 印出；fallback /tmp 時會明確提示。
仍未訓練 LoRA；若配對案例仍持續不遵從，下一步應建立正確 action/state 的訓練資料並訓練，
而不是把更多文字當作可靠操作或得分保證。
