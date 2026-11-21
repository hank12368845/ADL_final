# 固定step84的一場完整市場診斷

2026-11-20。使用者拍板固定step84，暫不訓練、不改persona、不換部署檔，量實際surplus與失誤成本。
step84在小題的收束／部分安全較穩，仍不是已證明的全面市場勝者。這輪只作固定基線的實戰診斷。

## 執行

在repo根目錄貼上：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_step84_market.py --gpus 0,1
```

GPU由使用者啟動。指令會啟動vLLM→跑一場→停止自己啟動的服務→自動產生報表，不接訓練／其他市場。
不填NEW_RUN；固定官方 `training/runs/20261119-213016_buyer_official_u3csnqyh/checkpoints/step_0084`。
啟動前核對兩份權重SHA與thinkingON step84標記，原persona逐字對照153410裸底座對照保存的原文。
在新結果目錄產生隔離agent／config，直接讀原step84權重，**不改agents下任何persona、agent.yaml或checkpoint**。
舊 `evaluate_buyer_market.py` 預設綁歷史step32與兩個版本的多輪換，這次不要用那條指令。

## 規模與正式規則

- 5個toy TA seller、10個buyer：我們p1／p2／p3加7個toy TA buyer。
- 1場，5round×6turn，共30turn。全市場450次行動；我們每persona30次、合計90次。
- 公開24商品，完整拷貝economy.yaml的market及phase2覆寫，budget_extra0.15、history6、每round8192輸出tokens含thinking。
- 我們Thinking ON；vLLM temperature0.7、top_p0.95、max_tokens8192、context64000、官方qwen3_coder／qwen3 parsers與原LoRA rename。
- base／tokenizer revision固定 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，沿用已下載模型，不改觀測排版或提示詞。
- 固定seed0，不跑底座場、不跑五次rotation、不跑三個economy。max_workers15只增加本場並行服務請求，不改同時觀測與隨機執行規則。
- 保存每回合完整observation、messages、原始reasoning／text、工具／結果、server／tokens與最終交易。

## 雙卡與進度

逐卡查當前剩餘顯存，扣2GiB餘裕，至少14GiB vLLM啟動預算；兩卡可用則兩個獨立副本。
三個學生persona共用同一step84 adapter、固定server0；TA personas按角色／thinking的生成成本先驗分流，並驗證實際15個agent的endpoint。
不宣稱兩卡每回合完全均衡或一定加速；結果actual_server_load列各endpoint實際呼叫、輸入與生成tokens。
有人佔用導致一卡不合格時醒目列skip，使用可容納的卡，不終止他人工作、不動態吃滿剩餘VRAM。
檢查結果磁碟至少5GiB與runtime cache至少5GiB；不足時按既有規則明示/tmp fallback，勿刪共用HW3 interpreter。
預設ports8000／8001；若已有服務佔用可明確加 `--port 8100`，不終止別人的服務。

每5秒終端進度條顯示0～450、round／turn、p1／p2／p3各0～30、elapsed、actions/s與ETA，保存progress.log。
載入／warm-up無速度時顯示estimating。歷史同尺寸市場約16分鐘，本輪可先預留20～40分鐘，以實際ETA為準，非時間保證。

## 三份帳：不能混成確定扣分

1. **已成交直接損益**：原引擎raw surplus、總支出、效用、已持有再購次數／金額、零邊際效用支出。
   逐單位依原引擎排序與rounding重播，與每筆trade.utility、最終spent／utility／cash／score核對。
   high-after-low可增加效用，所以再購總支出不直接當損失。第一筆qty>1的額外單位也逐單位列出。
   surplus = 有利單位淨收益 − 負收益單位損失 = 非零效用單位淨收益 − 零效用支出。
   這些金額已包含在surplus，不再額外扣一次，也不將重疊欄位相加。
2. **回合損失**：insufficient_cash、無工具、最終length截斷、length retry、round_budget_used、錯掛牌／offer及no_stock分類。
   沒成交的現金不足／截斷，直接交易支出為0；無法只憑單場紀錄精確算「因此少賺多少」。
   真正觀測有貨且同tick被買走、要求qty超過觀測庫存、亂選未掛牌、offer無庫存分開列，不把所有no_stock怪給模型或競爭。
3. **可見機會／餘額敏感度**：只用當時買家可見的board／offer／效用／grade，計算最低品質下仍為正的機會。
   cash_gap_snapshots列「若保留先前零效用支出，在這個原盤面多買得起什麼」。
   這不是重跑市場，不保證能成交、不代表確定regret，也不把重複出現的商品跨turn加總為損失。

## 搶快淨收益與後半段穩定性

逐persona看surplus、buyer名次／10、有利單位收益與錯單損失；by_kind列scarce等商品的首件／總件數、獨特商品與surplus，by_round列成交與淨收益。
這可回答本場是否整體賺得多於賠，不能單憑一場宣稱激進策略因果上比保守策略更好。

turn1–3與4–6各15個行動，並列各slot5個行動：cash錯誤的購買嘗試分母、截斷的模型決策分母、實際input tokens／history、剩餘round budget。
history不隨round清空，turn位置不是上下文長度的替代；資金下降、庫存與競爭也會影響後半段表現。by_round記錄每輪8192用量與被迫wait。
保留官方length_retries與基礎設施行為，不做自訂工具修復；報表另外列官方policy fallback／infra錯誤。

## 輸出與回讀

`results/basic_checks/<timestamp>_step84_market_*`（或終端明示/tmp位置）：

- **先讀market_loss_review.md**，數字／分類／完整逐回合帳在market_loss_review.json。
- run_market_*/standings.md、markets/*/final.json／scores.json為原引擎結果；events.jsonl含完整模型草稿與觀測。
- runner.log、progress.log、wrapper_result.json、resource_plan.json記錄執行、進度與雙卡路由。
- suite.json保存來源／引擎／persona／adapter的SHA、正式規則與本場座位；buyer/為原persona隔離副本。
- mock為CPU管線測試，valid_real_diagnostic必須False。基礎設施錯誤另標，不把它當正常模型實力。

只重讀已完成報表、不碰GPU：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_step84_market.py --report 實際結果目錄
```

`--prepare-only`不查GPU、不啟動服務；`--mock`只跑原引擎CPU假模型，不能當策略成績。
此wrapper沒有中途續跑；失敗保留全部log，先查原因，不自動重開或增加場次。
讀完本場戰損才安排下一次訓練；單場公開toy對手不能當最終高分保證。
