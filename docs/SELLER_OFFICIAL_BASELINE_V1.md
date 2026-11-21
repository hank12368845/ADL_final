# Seller 官方資料 LoRA 基線 v1

本輪已由使用者完成：run135008_seller_official_9d95_ubx、46steps、約45分鐘，詳[完成後獨立結果](SELLER_OFFICIAL_BASELINE_V1_RESULTS.md)。step46有貨報價3→9/9、截斷8→0，但全空庫存仍掛價，尚未驗profit；不要重跑下方訓練或直接續epoch。另已釐清無庫存set_price／過長private_state在引擎的實際處理，不能把開發守則當工具拒絕。

使用者已批准先建立 Seller 第一版 LoRA，保留 p3 v2 收益候選、p1 v1 對照。此輪固定現用 v2 三 persona 作訓練與主要驗收輸入，p1 v1 只在額外診斷題作對照，沒有修改 agents 內的 persona／權重。由指定 Qwen 原始底座建立新 Seller adapter，不接續 Buyer 或 TA 權重。

## 執行

在專案根目錄執行，一條命令包含資料核對、等待資源、GPU 反傳預檢、底座評估、完整一遍訓練，以及三個存檔的自主推論評估：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_official_seller.py \
  --gpus 0,1 --wait
```

預設使用兩張可用卡，只有一張符合訓練預算時會明示單卡；每卡剩餘顯存扣 2 GiB 後，至少需 18 GiB 訓練預算。不是先前市場 serving 的 14 GiB 門檻，也不把兩張卡 free 相加。等待顯示資源與已耗時、ETA unknown；不是資源預約，不終止別人的程序。啟動時顯存又被占用仍可能預檢失敗，停止並留 log，不盲目重啟。

程式自行在 HF 子程序啟用既有 FLA／causal-conv1d 隔離 overlay，無須 source，也不修改 vLLM 環境。CPU 已確認 Transformers 選到這些實作；GPU 數值／反傳／速度尚待本次預檢，不能先宣稱已加速多少。保留已驗證的 gradient checkpointing／2048-token MLP 分塊，完整 attention 不切段。

GPU 由使用者啟動，Codex 僅完成 CPU 準備及測試。首次 JIT、載入和反傳預檢 ETA 顯示估算中，後續訓練／自由生成進度有已完成／總量、elapsed、速度及 ETA。總時間未實測，不保證固定分鐘數。

## 資料與訓練配方

- 官方 `miulab/ADL26W-final-train` 固定 revision `fa0f94cff3eeed5bc54da597ad5c2a45ef491e45`，完整審查 1208 seller 軌跡／38772 決策。資料及底座已快取，不需重新下載。
- 只保留原生有 reasoning、恰好一個成功工具、當前有庫存、所選 grade 售價高於成本的原始答案。移除舊 horizon、不相容 schema、超額 offer、舊多件需求 offer、來不及生效的最後掛牌，以及超過 384 tokens 的教師思考。教師 completion 沒有改寫或合成。
- 將舊 system 改為現行官方 Seller 規則／工具＋現用 v2 persona；私人成本、庫存、商品價格原值保留。商品敘述使用官方 catalog 一行摘要；沒有在部署加 formatter 或額外答案。
- 完整市場 group 分 train／validation／test，角色、persona、同世界不同軌跡不跨 split。最終 train **363**、validation **14**、test **35**。上限 1024 並非硬湊 1024；篩選後有 414 筆候選，未將短資料複製擴增。
- Train：227 set_price、92 make_offer、28 send_message、12 post_to_room、4 wait；36 個獨立 public-market groups、119 個來源軌跡。p1／p2／p3 樣本為 129／155／79。正報價佔約 88%，不將低 loss 當學會策略。
- 全為 6／12／24 商品的舊資料世界，train 中 24 商品只 1 筆。稀少到不足三個獨立市場的形狀留 train，不拆同一市場偽造 holdout；官方 val／test 沒有 24 商品世界。不能聲稱已覆蓋正式全部情境。
- 完整當前 observation 不截字；最多保留 2 個歷史 turn，超過長度時只刪最舊完整 turn。max input+target 8192，train 平均約 7029 tokens；正式市場仍由 TA harness 提供最多 6-turn history、64K context，此訓練取樣不改市場規則。
- Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，bf16、rank16／alpha32、dropout0、僅 language layers、PEFT safetensors、no DoRA。
- Thinking ON；一般完整 assistant completion CE，prompt/history 不算 loss，不加 tool-name 權重或新檢核格式。AdamW LR5e-5、warmup3 steps、cosine decay、global batch8，單／雙卡都保持相同 global batch。
- 一個完整篩選子集 epoch：**46 steps**，363 個不同 train 樣本全覆蓋；尾 batch 補 5 個既有樣本對齊 batch，共 368 曝光。保存 **12／23／46**，不自動加 epoch。

## 驗收與輸出

GPU 最長樣本與最長 target 各做真實 forward/backward，確認 checkpointing 啟用且 loss 有限，這時尚未 optimizer 更新。Step0 新 LoRA 的 B 初始為 0，與底座等效。然後訓練，三個 checkpoint 都自由生成原始答案，不提供正確草稿、數字或工具前綴。

每版 **25 calls**：6 個官方未見、5 個前段已見、12 個現行引擎盤面，以及 2 個 p1 v1 對照，共底座＋3 checkpoint＝**100 calls**。現行題包含可掛牌、有買家詢價、無庫存、8 個未結 offer 的情況；維持三 persona 分項原文。12 個引擎題是小型 development probes，不是市場 surplus。

另外計算官方 validation completion NLL／工具分項；使用原生 Qwen template 與 vLLM qwen3_coder parser 解析 HF 自由生成。小題 greedy1024輸出上限不是正式比賽 token 設定；截斷須獨立標示。這裡 `valid` 是 schema／庫存／正毛利等開發條件通過，不能冒充全部是 engine 明文違規；例如最後一回合再掛牌可能合法但無後續買入機會，資料不教此動作。

若自主報價捕捉或操作通過較底座少 4 題以上，或原本能報價卻變成完全不報價，正常停損，保留 checkpoint。選模同時看自主報價、庫存／成本等錯誤、截斷與未見題，不單看 NLL；仍可能沒有合格候選，不自動改鬆門檻或再訓練。

結果：`training/runs/<timestamp>_seller_official_<suffix>/`，重點檔案：

- `seller_training_review.md`：先看這份。
- `native_step_0000/0012/0023/0046.json`：每題原始思考、工具、錯誤與分項。
- `checkpoints/step_0012/0023/0046/`：adapter、optimizer／RNG 與存檔完成指紋。
- `runner.log`、`metrics_rank*.jsonl`、`samples_rank*.jsonl`、`preflight_rank*.json`：過程、進度／ETA、來源曝光與加速實作／顯存證據。
- `finished.json`：診斷候選與是否停損；不是正式高分或市場 profit 驗證。

同一指令已自動評估存檔。跑完不自動市場、部署或改 agents 權重；下一步固定 persona，比較候選與 starter 在一場新 seed、正式 5×10／30 turns 的實際 profit，保留未見來源與成功／退步的證據。

中斷可接續實際資料夾：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_official_seller.py \
  --resume training/runs/實際_seller_official_資料夾 --gpus 0,1 --wait
```

須維持原 GPU 數量，保留 optimizer、scheduler、逐 rank RNG、固定 batch 與來源指紋；若原單卡而現在兩卡都合格，請以 --gpus 指定一張，避免改變 exact-resume world size。完整存檔才算完成；無存檔則從底座重來。已完成 run 只重寫 CPU 報表，不重訓。

## 重現與 CPU 證據

首次重現先依 `training/download.sh` 取得固定底座與官方資料，依 `docs/TRAINING_FAST_KERNELS.md` 安裝隔離加速套件；執行上述命令，資料不存在時會以 CPU 重建。配方 `training/configs/seller_official_v1.yaml`，資料處理 `training/official_seller_data.py`，訓練 `training/official_seller_train.py`，啟動 `training/train_official_seller.py`。

9 項 CPU 測試已通過：來源 completion 原樣、私有數字與現用 persona、整市場隔離、完整 epoch 曝光、native token loss／end-of-turn、引擎盤面限制、已見題覆蓋、全 wait 退化門檻及 CPU 入口不啟 GPU。證據 `/tmp/adl_seller_official_tests.log`、`docs/SELLER_OFFICIAL_BASELINE_CPU_REVIEW.json`。CPU 加速 dispatch 檢查不是 CUDA 數值／吞吐驗證。
