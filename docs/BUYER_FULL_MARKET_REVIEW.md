# Buyer完整市場評估：用surplus檢查目前候選

**歷史方案：本頁指令綁舊step32與兩版本多輪換。2026-11-20現用step84的一場市場診斷請改讀docs/BUYER_STEP84_MARKET_DIAGNOSTIC.md，勿直接用本頁舊預設。**

2026-11-19。使用者要求先測目前較強版本的真實市場實力、找出需要加強之處。
這是推論評估，不訓練新LoRA。GPU由使用者啟動；Codex已完成CPU設定、測試與mock流程。

## 比較與規模

- candidate：board-repair-v1 step32，`training/runs/20261119-134251_buyer_operations_50lhkufb/checkpoints/step_0032`。
  這是目前較穩的操作對照，尚未證明是市場最高分；ID／symmetric退化分支不加入此輪。
- base_on：同一Qwen3.5-4B、完全相同三個訓練persona、thinking ON，直接用底座、無adapter。
- 兩邊同樣seed0、三個學生buyer、七個toy TA buyer、五個toy TA seller；各自獨立市場，避免兩版本在同場改變對方機會。
- 直接使用economy.yaml的市場規則與phase2覆寫：公開24商品、15%budget_extra、完整5round×6turn、每round8192tokens、6turn歷史。
- 每個版本完整五次庫存輪換，共10場、4500次agent actions；每persona在每版本有5場、150次行動。
- 同ID、相同對手／persona／規則／seat／rotation／seed，只有三個學生buyer的LoRA不同。之後互動軌跡會因模型決策分岔。
- 公開toy TA與單一public economy不能推論最終三economies成績，也不能宣稱比較過所有歷史checkpoint。

原始agents與提交checkpoint不覆寫；在新的suite目錄產生隔離agent/config，保存來源、權重與輸入SHA。
persona取候選訓練run的三份原檔，與其資料manifest核對，避免誤用現在example_student中的另一版長persona。

## 啟動

在repository根目錄：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_buyer_market.py --gpus 0,1
```

預設先candidate的五輪換，再base_on的五輪換；每邊都使用兩張可用卡，沿用已驗證的TA成本分流，
逐卡按剩餘VRAM扣餘裕，顯示實際agent routes，保存resource_plan與實際server呼叫／token負載。
若只有一卡符合顯存預算，既有basic wrapper會顯示skip與實際選卡；不終止其他人的工作。

啟動前檢查結果／cache空間；編譯快取使用final runtime cache，沿用basic wrapper的不足空間處理。
終端同時顯示場內round／turn、整個版本的actions，以及跨兩版本的4500-action整體進度、elapsed、速度與ETA。
載入／warm-up尚無速度時顯示estimating。整體ETA依實際已完成actions／累計耗時估算，兩版本速度不同時會更新，並非時間保證。
舊完整5×10單場約16分鐘，這次10場可先預留約2～4小時；新thinking、分流、庫存耗盡後輸出及服務啟停都會影響時間。

輸出在`results/basic_checks/<timestamp>_buyer_market_review_*`，保存：

- `suite.json`：候選／來源／規則／完整paired plans／persona與生成輸入SHA。
- `state.json`、`launch.json`：已完成版本、實際啟動選項與耗時。
- `suite.log`：兩個版本的完整wrapper輸出與整體進度。
- `candidate/output/.../run_market_*`、`base_on/output/.../run_market_*`：原引擎standings／score／各rotation對話與events；各basic session有runner.log、progress.log、vLLM logs及resource_plan。
- `market_review.md`／json：candidate完成就先產出部分報表，兩邊完成再給完整對照。

如果只想先看每邊一個完整30turn市場，可以明確加`--single-market`（共2場、900actions），
但預設長測保留全部五次輪換。

## 讀結果

第一個讀`market_review.md`，逐persona看：五輪換平均／最低／最高surplus、buyer名次、
candidate相對base的同輪換surplus差與勝／同／負。原引擎surplus是主指標，pooled standings或normalized不代替它。

同時看以下原因，才能決定是否值得再訓練：

| 觀察 | 優先檢查 |
|---|---|
| no_stock但當前row真的有貨，且同tick成交耗盡 | 搶購碰撞、標的與時機選擇 |
| seller/product不在當前有貨board，或offer_id不存在 | 當前來源綁定、歷史／其他商品混用 |
| insufficient_cash | 目前cash與總價核對 |
| wait時仍有可見正surplus下界選項 | 過度保守、選項掃描、token耗盡等待；不能當成保證可成交的損失 |
| 零效用支出、低品質負收益 | 重複購買與品質策略；low商品也可能有正surplus |
| 推理截斷、round_budget_used、無工具 | thinking長度、工具輸出與8192round budget |

JSON保存每persona動作分布、拒絕原因、現金、成交件數、實際server負載，以及各問題的tick／當前board／offer／reasoning。
品質損失與零效用支出已反映在surplus中、彼此可能重疊，不再扣一次。
可見正收益下界只掃未持有、可支付且有效的board／offer，variable-quality以0.4倍價值保守估算；
未使用隱藏grade，不是全局最優策略、事後regret或保證成交機會。

## 中斷與重讀

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_buyer_market.py \
  --resume 實際suite目錄 --gpus 0,1
```

`--resume`保留已完整完成的版本，未完成的版本從頭重跑；不是從市場中途checkpoint接續。
來源／生成persona/config改動會被拒絕。mock與真模型不能混合；檔案都保留以便查錯。

CPU重新讀報表：`training/evaluate_buyer_market.py --report 實際suite目錄`。
CPU準備：`--prepare-only`；CPU管線mock：`--mock`，報表明確標記有效真模型評估為False。
CPU證據見`docs/buyer_public_market_cpu_review.json`；尚無這輪真實GPU市場結果。
