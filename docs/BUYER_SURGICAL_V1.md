# Step84局部修補：CPU資料與配方審查

2026-11-20，依使用者要求保留step84成功行為，只修來源配對與迴圈。本輪已由使用者完成CUDA訓練及306題評估，step10／20均未過門檻；**目前不直接重跑或續訓**，最新見[BUYER_SURGICAL_V1_RESULTS.md](BUYER_SURGICAL_V1_RESULTS.md)。以下保留重現方法。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_surgical.py --gpus 0,1
```

同一指令包含雙卡資源與最長樣本反傳預檢、父step84的102題自由生成、20步訓練、step10／20各102題評估，共306次決策。
兩卡各自分配51題／模型，訓練DDP每卡2筆／步；原step84、persona、engine與部署不改，不自動加場或續epoch。

## 已備妥資料

位置：`training/data/buyer_surgical_v1/`。CPU準備程式為`training/prepare_buyer_surgical.py`；既有實驗與原三persona均未修改。

| 組別 | 不同樣本數 | 訓練曝光數 | 內容 |
|---|---:|---:|---|
| 原step84成功重播 | 43 | 56（70%） | 真實市場正surplus成交，原messages、reasoning、可見文字與原工具參數保留 |
| 局部修補 | 24 | 24（30%） | 13筆真實失敗修正＋11筆引擎控制盤面 |
| 合計 | 67 | 80 | 20步，每步global batch4 |

每5步恰好14筆重播／6筆修補；43筆原成功樣本全部至少曝光一次，13筆額外重播如實記在manifest，不宣稱56筆獨立成功。
三persona曝光分別24／30／26筆，p2因本場錯配較多而有較多修補。
70/30指樣本曝光比例；原重播／修補target tokens為16327／5352（約75.3%／24.7%），不是假設token或梯度比例也相同。

原成功資料來自`20261120-160558_step84_market_st8d6o1p`的一場正式尺寸市場，43筆全是buy_listed，沒有成功accept_offer。
只保留實際單位效用減支出為正、當前來源有效、現金足夠及符合原persona冷卻規則的成交，排除5.40零效用重買。
「成功成交」不表示每次選品都最優，也不能代表step84九成思考都正確。

## 24筆修補的範圍

| 類型 | 筆數 | 處理方式 |
|---|---:|---|
| 真實seller／product配對錯誤 | 11 | 保留原觀測與歷史，只改錯誤草稿及最後工具，使用同一當前row的ID |
| 真實迴圈 | 2 | 不回灌整段重複文字；修正當前可交易候選並正常結束工具 |
| 有效offer、不在board | 5 | 明確使用accept_offer，不因沒有公開掛牌而拒絕 |
| 首列虧本、後列有利 | 3 | 繼續比較當前board，買後列有利商品 |
| 冷卻中，有其他好商品 | 1 | 避開剛失敗的配對，轉買另一個有效候選 |
| 冷卻到期 | 1 | 原引擎失敗→等待一回合→重新檢查新掛牌，才買回相同配對 |
| 冷卻中，無其他好商品 | 1 | 合法wait；避免把所有失敗都教成直接購買 |

冷卻、現金、stock、工具通道與觀測欄位都做CPU驗證。冷卻判斷兼容正式harness把結果放在history/tool message的情況，不只找last_action_result。
控制盤面不直接覆寫現金／持有欄位，教師工具經原引擎執行成功與收益核對。board最多29列，避免全部只剩幾列的簡單題。
原43筆成功reasoning逐字保留，原native content與private_state也保留；不將「原樣保留」誤稱為每一個記憶欄位都已修正或正確。
修補草稿採自然語句，不新增JSON／CASH等思考格式，不以150／300等上限迫使全部思考變短。
原成功think為25～821tokens；不強迫收斂成上一輪90tokens模板。正常native think→工具→EOS仍須驗證。

## 凍結的下一輪配方

`recipe.json`固定官方step84、原三persona、Thinking ON、rank16／alpha32；不從上一輪step24／48接續。
20 optimizer steps、global batch4、雙卡各2筆／步，fresh AdamW peak LR5e-6、warmup2、cosine，保存step10／20。
正常completion CE、prompt mask，不加決策token或檢核格式的特殊loss，不自動增加epoch。

完整input＋target最長24432tokens，80次曝光合計1422895tokens，約為上一輪384筆token總量的45%。
這是token預算，不是時間保證；長上下文的成本不同。最長樣本反傳由使用者啟動後先驗證，不能預先宣稱兩張4090已確認可裝下。
保留全部history，訓練長度達16000tokens時使用PyTorch save_on_cpu暫存反傳activation，配合gradient checkpointing減少顯存壓力。
不改資料或loss；CPU微型autograd對照通過，完整Qwen/CUDA的相容性與顯存仍由真實預檢確認，預檢失敗就停止。
此處的20步只是有上限的候選，不保證消除loop或提高surplus。此前同時改過資料／格式／步數，無法斷言所有退步都只由思考格式改變造成。

## 驗收與資料邊界

本次使用者已授權將這一場市場用於訓練，故舊6筆真實市場回歸已標成`training_market_seen`，不可再當未見驗證。
凍結原36題、新來源24題、壓力18題及6題舊合成probe，另加12題不同來源的offer／首列跳過／冷卻對照，合計102題。
12題新增控制盤面使用官方validation的6來源群組，與本輪train來源完全不重疊；各題均有persona原文，雙卡路由51／51。
這些是development驗收，並非封存test，亦無新市場surplus。
後續驗收必須同時檢查：原有利交易捕捉與選品的可見gain、來源／通道錯誤、截斷與尾部token消耗、cash／held回歸。
不能只因合法率上升、沒有迴圈或買了低收益小商品就晉級；真實市場收益需新seed／市場確認。
本輪沒有新模型推論結果；不能沿用上一輪改壞權重的成績當本輪結果。

## 可重現的CPU操作

原始資料與step84權重仍依既有官方流程準備；本程式完全離線讀現有來源。
預設資料已生成，不覆蓋。需要重現到另一個新資料夾：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/prepare_buyer_surgical.py --output-dir /tmp/adl_surgical_reproduce
```

只讀核對既有資料：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/prepare_buyer_surgical.py --audit training/data/buyer_surgical_v1
```

CPU測試：`python -m unittest training.test_buyer_surgical -v`，使用專案指定環境Python。
測試涵蓋原始草稿／輸入逐字保留、70/30曝光、冷卻紅線、offer無board、首列跳過、來源切分、102個實際API schema、native工具解析、資料竄改拒絕。
manifest記載來源／父權重／資料SHA、版本、重播次數與限制；這些CPU證據不等於GPU訓練有效。

## GPU流程與輸出

每階段按剩餘顯存扣2GiB重新檢查，兩卡各至少18GiB啟動預算，不終止其他人工作。開始前檢查結果與cache空間並明示fallback。
預設8000／8001，若已有服務可在啟動命令追加`--port 8100`。
載入有心跳，訓練／評估有完成數、elapsed、速度與ETA；首次進度前ETA顯示估算中。
結果在`training/runs/<timestamp>_buyer_surgical_*`，終端會印出完整路徑。

- `preflight.log`／`preflight_rank*.json`：最長樣本反傳與顯存紀錄。
- `eval_parent.log`／`train.log`／`eval_candidates.log`：階段log。
- `checkpoints/step_0010`／`step_0020`：adapter、optimizer、RNG、完成指紋。
- `evaluation_parent.json`／`evaluation_10.json`／`evaluation_20.json`：逐題思考、工具、錯誤及路由。
- **`surgical_review.md`／`.json`**：先讀此處；保留交易捕捉、現金／持有、來源與收束門檻，候選可以為None。

新報表增加每題所選交易的可見最低品質正收益合計，避免合法買入低毛利小商品就冒充改善；此數字加總不同snapshot，不能當真實市場surplus或regret。
舊binding題庫中保留的new_listing輸入仍不變，但與persona衝突的立即重試工具會被標為persona_cooldown；可以改選當前其他有利掛牌。
三模型都重新評估，不拿上輪較寬判準或不同GPU路由的cached數字混比。
程式不自動改agents，不自動部署或開完整市場。

中斷後若有完整step10／20，可用`training/train_buyer_surgical.py --resume 實際run目錄 --gpus 0,1`恢復；
若訓練完成只需補評估，用`--evaluate 實際run目錄 --gpus 0,1`。保持相同來源、版本與卡路由；沒有完整checkpoint則開新run。
`--cpu-check`只讀資料與實際API schema，不查GPU、不啟服務。啟動器CPU測試在`training/test_buyer_surgical_runner.py`。

## 首次長樣本反傳OOM修正

`20261120-202539_buyer_surgical_om5lmi7k/preflight.log`顯示兩卡在最長24432tokens的反向重算均OOM，錯誤落於Qwen3.5 MLP的PEFT輸入轉fp32。
每卡上限約21.07GiB，該程序已用約21.45GiB；反傳尚未完成，也尚未執行optimizer step。不是torchrun parser或資料格式錯誤。
save_on_cpu降低保存activation的成本，但仍不足以限制整層MLP重算的瞬時峰值。

修正僅作用本地trainer：Qwen3.5 tokenwise MLP以2048tokens分塊，內層non-reentrant checkpoint，保留完整attention、所有輸入／target與原loss。
不改LoRA dtype／rank／資料、全域batch、正式engine或vLLM；保存的adapter參數名稱與格式不變。設定`PYTORCH_ALLOC_CONF=expandable_segments:True`以減少碎片。
CPU測試使用安裝版本的原生Qwen3_5MLP＋PEFT，核對完整／分塊輸出、input與LoRA梯度、外層checkpoint＋CPU offload的巢狀反傳，以及bf16 base／fp32 LoRA；不是CUDA顯存已通過的證據。
完整Qwen/CUDA仍由使用者啟動後的預檢確認；不能只因CPU通過就保證不OOM。
原失敗run沒有checkpoint，保留log與來源指紋，不用--resume；重新執行本文首條原命令建立新run，資料不需重做。
