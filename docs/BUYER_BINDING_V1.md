# Buyer掛牌配對與思考收束：step84接續訓練

2026-11-20。使用者已完成本輪GPU訓練與驗收，**step24／48均未通過，暫停重跑或延長**。最新結果與資料衝突見[BUYER_BINDING_V1_RESULTS.md](BUYER_BINDING_V1_RESULTS.md)。下列保留本輪重現方法，不是目前建議的新工作。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_binding.py --gpus 0,1
```

這一條包含資源檢查、最長樣本反向傳播預檢、父模型90題、48步訓練，以及step24／48各90題評估，合計270次模型決策。
預設使用8000／8001；若已有服務，可追加`--port 8100`，使用8100／8101。不要終止別人的程序。

## 固定訓練設計

- 父權重：`training/runs/20261119-213016_buyer_official_u3csnqyh/checkpoints/step_0084`。Qwen/Qwen3.5-4B revision、rank16／alpha32、Thinking ON均固定。
- 原三persona全文、正式observation／history／原生工具模板保持不變。不加檢核格式、不改正式harness，不依賴1200 tokens上限。
- 384筆、global batch8、48步完整一遍；每batch為5 buy_listed／2 accept_offer／1 wait。資料均只曝光一次，兩張卡各處理4筆／步。
- Fresh AdamW、peak LR `1e-5`、warmup4步、cosine decay、gradient clipping1。父模型只載權重，不載其舊optimizer；本輪中斷續跑則恢復本輪optimizer與RNG。
- 正常assistant completion CE，prompt不計loss；完整學習短think、關閉think、工具及EOS，不再加決策token特殊權重。
- 每份教師草稿最多300 tokens，實際最長90。完整input＋target平均8244、最長13728 tokens，保留長history，不截斷輸入偷降低難度；安全上限20480。
- 保存step24／48，兩個都評估；不自動加epoch、不部署、不接完整市場。

## 資料來源與用途

`training/data/buyer_binding_v1/manifest.json`凍結來源、persona、父權重、版本與資料SHA。
來源為官方資料的商品／效用世界，按官方既有整市場切分，取16個train來源群組與6個validation來源群組，兩者不重疊。
在現版引擎下重新構造並驗證狀態轉移，**不是直接重放官方舊版horizon的原始行為軌跡**。

288筆針對多賣家配對、同商品不同seller、相似seller ID、舊history與當前row衝突、no_stock後換seller／商品、not_listed後新掛牌可重新購買，以及有效offer／沒有有效交易。
另外96筆為官方世界的正常成功board／offer演練，保留交易能力。多筆正收益候選避免只學「盤面唯一便宜貨」。
現金與持有來自真實引擎交易；歷史混合成功買入、接受offer、訊息、評論與等待，不是連串wait前綴。教師動作逐筆經引擎驗證成功且正收益。

## 同條件自由生成驗收

每個模型90題：

| 題組 | 題數 | 用途 |
|---|---:|---|
| 原36題 | 36 | 凍結現金／持有與交易捕捉回歸 |
| 新來源配對驗證 | 24 | 未進train的來源世界，配對與失敗後更新 |
| 壓力題 | 18 | board／offer等額、差一分錢、零現金、二選一與品質／升級 |
| 訓練已見題 | 6 | 區分沒有學會與泛化失敗 |
| 本場真實紀錄 | 6 | 兩個loop、錯seller、連續失敗及最後重買；不進train |

父step84、step24、step48都重新使用正式vLLM、temperature0.7／top_p0.95、原policy與8192每round剩餘額度。
不拿舊HF greedy／1024上限的結果直接當新對照。每模型兩卡各45題，逐題固定路由，原始草稿、工具、token數、錯誤與request紀錄均保存。

按persona檢查正例捕捉、負例安全、現金／持有與截斷／loop；新配對須有改善且原能力不退步，不能靠全wait過關。
同snapshot可有多個合理交易，不強迫只選教師同一商品。品質的保守界線是策略設計，不冒充真實品質機率；先low後high可能有正邊際效用。
這是操作與收束驗收，**沒有完整市場surplus或正式高分證據**。六個已看過的真實案例也只算回歸，不能宣稱未見市場泛化。

## 資源、進度與結果

兩卡DDP訓練；推論使用兩個獨立vLLM副本。每一階段重新讀取剩餘顯存、扣2GiB餘裕，每卡至少18GiB啟動預算，不足就停，不會動態霸占別人的顯存。
開始前檢查結果磁碟與cache；預設輸出`training/runs/<timestamp>_buyer_binding_*`，空間不足會明示替代位置。
載入／warm-up有心跳；訓練及評估顯示完成數、elapsed、速度、ETA。第一批完成前顯示估算中；270次決策不代表270次HTTP呼叫，原policy可重試。
整體時間取決於卡速、長history與生成loop，以實際階段ETA為準；不能把48步當成數分鐘內必完。

主要檔案：

- `train_config.json`／`batch_plan.json`：凍結配方、來源、版本、路由與資源快照。
- `preflight.log`／`eval_parent.log`／`train.log`／`eval_candidates.log`：各階段原始log。
- `preflight_rank*.json`：最長樣本反傳與顯存紀錄；`metrics_rank*.jsonl`／`samples_rank*.jsonl`：loss與實際樣本曝光。
- `checkpoints/step_0024`／`step_0048`：adapter、optimizer、RNG與完成指紋。
- `evaluation_parent.json`／`evaluation_24.json`／`evaluation_48.json`：逐題草稿與錯誤。
- **`binding_review.md`／`.json`**：優先閱讀，列出門檻與候選，候選可能為None；不自動修改agents。

若中斷且已有完整step24／48，用實際run路徑續跑：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_binding.py --resume training/runs/實際run目錄 --gpus 0,1
```

只補已完成checkpoint的評估：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_binding.py --evaluate training/runs/實際run目錄 --gpus 0,1
```

勿把範例路徑原樣貼上；若首24步尚未存檔即中斷，只能開新run。恢復時來源、資料、版本與卡路由必須一致。
單純CPU重查可用`--cpu-check`，不查GPU、不載模型；CPU測試含模擬client／controller，只證明流程與評分邏輯，不能證明LoRA有效。

## 首次啟動的torchrun參數修正

`20261120-184341_buyer_binding_3_z_1z9n/preflight.log`顯示`ambiguous option: --run could match --run-path, --run_path`。
這是torchrun的參數縮寫解析誤收子程式`--run`；模型尚未載入、訓練尚未開始。
啟動器已在torchrun選項與training script間加入`--`，CPU測試直接使用目前安裝的`torch.distributed.run.parse_args`驗證preflight／train兩種命令。
此失敗run沒有checkpoint，不用`--resume`；重新執行本文首條命令建立新run，資料與父權重不需重建。舊log與來源指紋保留作追蹤。
