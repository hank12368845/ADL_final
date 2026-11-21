# Buyer Operations v2.2：只改決策監督的短對照

目的：同一個 offer，只改目前現金，模型能否從 `accept_offer` 切換為 `wait`，
同時保留足夠現金時的正收益成交。v2.1 final 在 cash probe 中買不起的 offer 仍接受47/48次，
零現金仍接受11/12次，所以現在不直接延長訓練或部署 final。

## 唯一改動與固定條件

原目標：`mean(completion NLL) + 1.0 × mean(tool-name NLL)`。
本版目標：`mean(completion NLL) + 1.0 × first-tool-name-token NLL`。
完整 completion 仍受監督，沒有刪掉參數或記憶的 loss。

官方模板中的 buy_listed／accept_offer／wait 名稱分別為3／2／1個 token，
且三個名稱的首 token 不同。原額外 loss 對首 token 的係數分別為1/3、1/2、1，
本版均為1。這是監督位置與強度的單一對照，不是 pairwise contrastive loss，
也不保證改完就會比較現金；若只增加無條件買進，成對指標會揭露退化。

- 完全重用 v2.1 原資料與官方 prompt、persona、tools、target，沒有新增資料或重做 label。
- 每一步8筆、各rank及micro的sample ID順序與已完成v2.1 run逐筆相同。
- 固定seed、r16／alpha32、LR1e-5、weight decay0.01、warmup2、16steps與global batch8。
- 從底座新建 LoRA，不從任何舊 step8 warm start；thinking OFF。
- 每4步存檔；完整名稱平均 NLL 與首 token NLL 均記錄，避免不同目標的 loss 混稱。

歷史控制為 `training/runs/20261118-233240_buyer_operations_pa514zcv`。
CPU會核對該run的配方、資料manifest、batch plan、兩卡實際sample traces，以及對照權重fingerprints；
配方或資料漂移會拒絕啟動。這是一次歷史控制對照，不是多seed重複訓練，
不承諾CUDA逐bit重現，也不把效果差異當作已確認的單一根因。

## 先跑訓練

本機資料、控制權重都已存在，不用重新下載或產生資料：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_operations_v2_2.yaml --gpus 0,1
```

這會啟動兩個DDP worker，各microbatch1、累積4次、各64次樣本曝露。
兩張卡不夠放模型就停止，不自動退成單卡，以保留本次控制條件。
啟動前檢查磁碟及逐卡可用顯存、保留餘裕；不清掉他人的工作。
先前相同v2.1流程雙卡約16分25秒，v2約17分22秒，可用16～18分鐘作粗估，
仍以終端實際elapsed／速度／ETA為準。載入與preflight尚無速度時顯示估算中。

輸出：`training/runs/<time>_buyer_operations_*/`，有 `runner.log`、`metrics.jsonl`、
`train_config.json`、`batch_plan.json`、各rank sample trace、`checkpoints/step_0004`至0016及
`final_adapter`。不自動部署、啟動vLLM或接續評估。Ctrl-C會停止worker；
有adapter存檔，但沒有optimizer續跑功能，再啟動會新建run。

## 再跑現金正反例

將下列RUN_PATH換成訓練完成時印出的實際run：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_buyer_operations_v2_2.py \
  RUN_PATH --gpus 0,1
```

預先指定先看step8，三模型為：

| Label | 權重 | 用途 |
|---|---|---|
| v2_step8 | 舊v2 step8 | 保留高召回參考 |
| v21_control | v2.1 step8 | 相同步數的舊監督控制；原候選保留 |
| candidate | 新v2.2 step8 | 唯一改動決策監督的候選 |

32題 × 3 persona × 2 seed × 3模型 = 576 calls，兩卡規劃各288次。
每個現金正反例的兩側、三個模型都在同一張卡；不同pairs分流雙卡。
報表逐筆核對模型間prompt SHA及GPU，並確認pair只改現金、其他觀測與歷史相同。
模型服務、call完成數、elapsed、速度、ETA及log沿用已驗證wrapper。
任何服務失敗、缺資料、重複結果或對照fingerprint變動均拒收，不能拿來選模。

`--checkpoint final`改為v2.1 final與v2.2 final的16步配對；
`--checkpoint step_0012`比較雙方12步，舊v2 step8始終保留。
先看預先指定step8的cash結果，再決定是否投入其餘checkpoint，不一口氣擴大市場長測。
`--port 8100`可更換起始port；`--prepare-only`只驗證及凍結輸入，不查GPU、不啟動服務。

結果在 `results/basic_checks/<time>_buyer_diagnostic_*/`；磁碟不足會明示fallback到
`/tmp/adl_final_basic_checks`。先讀 `cash_decision_pairs.md`，詳細資料在同名JSON，
原始與評分後回應、路由、config及runner／vLLM log均保留。

## 如何判讀

主要指標是 `both_correct`：同一pair的正例合法捕捉交易，而且負例合法等待，兩者都正確才得1。
offer、board分開看，另拆persona、fresh／recorded、equal／above_one／wide／zero。
同時看正例捕捉、負例等待、unaffordable、trade precision與兩側都wait／都交易的比例。
不能以全wait、全accept或單獨loss下降宣稱成功；記憶格式另列，不直接當成交分數。

值得繼續的證據：相較相同步數v2.1，offer的both_correct增加、零現金／明顯不足錯買下降，
正例捕捉能保留，且沒有只靠某個persona或context掩蓋退化。
若還是全accept，或只是從全accept變全wait，本輪不能升級。
若有清楚改善，再做少量完整5×10市場確認surplus；目前診斷仍非策略或正式grade。

這套32題來自一個held-out validation市場；persona／seed是重複條件，部分variant輸入也相同。
576 calls不是576個獨立市場。test split仍保留；不把驗證題改造成新訓練資料。

## CPU驗證與重現

CPU檢查：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_operations_v2_2.yaml --cpu-check
```

loss與梯度對照、舊預設loss相容性、控制漂移拒收、原生API schema、三模型雙卡假worker、
完整pair計分、缺結果與prompt不一致拒收，均以CPU完成，沒有執行真模型。
證據見 `docs/buyer_operations_v2_2_cpu_review.json`。

新機器重現先依training/README下載底座與套件、生成v2及v2.1資料，
重現v2與v2.1控制run，然後修改本版config的comparison_adapter／control_run為實際路徑，
並同步控制config中的comparison_adapter，才能做相同配方的受控對照。
本輪沒有新增套件，最終checkpoint尚未選定。
