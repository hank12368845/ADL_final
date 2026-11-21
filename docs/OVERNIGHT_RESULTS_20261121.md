# 2026-11-21 夜間結果與續跑

Buyer 已完成官方 Phase 2 economy；Seller 只有 v1 完成，v2 在模型啟動時遇到剩餘顯存改變，尚無策略成績。Codex 本次僅讀取結果、CPU 核帳與修正排隊器，沒有啟動 GPU。

## Buyer：step10，15 場完整市場

來源：`results/basic_checks/20261121-000742_step10_economy_a2txwbqx/economy_review.md`、同目錄 JSON 及官方 run 的 standings。原三 persona、Thinking ON，3 economies × 5 rotations、5 sellers × 10 buyers、每場 30 turns；總計 6,750 行動，每個學生 persona 450 行動。06:45 正常結束，市場執行約 4 小時 28 分鐘，另含排隊與啟動時間。

| persona | 平均 surplus／場 | 最差場 | 零邊際效用商品總支出 | 截斷 | 現金不足錯誤 |
|---|---:|---:|---:|---:|---:|
| p1 | 112.09 | -58.90 | 371.33 | 5 | 0 |
| p2 | 125.99 | 27.55 | 277.88 | 8 | 0 |
| p3 | 136.26 | 58.14 | 413.52 | 4 | 0 |

本次最好的公開 TA buyer 平均 137.96；p3 接近它，但公開 TA 只是 toy agents，並非正式評分對手。三個 economies 此次仍是公開商品檔，不代表通過兩個 private 商品世界。舊版 student pooled raw 124.78／grade 1.866 不是正式 best-persona 成績，不能據此保證最終分數。

獨立逐單位核帳已對照 45 個 persona-market 的 cash／spent／utility／raw 及報表。零邊際效用支出合計 1,062.73，平均每 persona 每場 23.62，已扣在上述 surplus，不能再扣一次；避免它們可能釋出現金與回合，但不能把全額視為新策略必然增加的分數。p1 唯一負分場有 103.33 的此類支出。品質升級不一律當成零效用重買。

另有當前 board 找不到的配對 p1／p2／p3 為 50／47／41 次、截斷合計 17、額度耗盡等待 29。它們與同 tick 被搶走分開；機會成本不是已知確定損失。現金錯誤 0 不證明一分錢邊界已解。API／模型服務失敗為 0，不等於每個工具都合法。兩個服務均有實際負載，沒有只開雙卡卻全部打同端點。

保留 step10 為收益候選、step84 為基線；本次沒有同 economy 的 step84 對照，不能單憑這份報表宣稱較 step84 提升多少。Checkpoint1 前先完成 Seller，暫不加 Buyer 訓練。

## Seller：v1 完成，v2 尚未開始市場

來源：`results/basic_checks/20261121-005859_seller_personas_1f4wxek7/v1/independent_seller_review.json`。starter adapter／Thinking ON、seed4、完整 450 行動，耗時約 21 分鐘；並非 Seller 專項 LoRA 訓練成果。

| persona | profit | 售出 | 未售 | 主要行動 |
|---|---:|---:|---:|---|
| p1 | 175.27 | 23 | 24 | 23 次 set_price、7 次 wait |
| p2 | 238.21 | 33 | 17 | 28 次 set_price、1 次訊息、1 次 wait |
| p3 | 171.00 | 26 | 23 | 30 次 make_offer |

三者無工具拒絕、截斷或額度等待，成交均 high grade。成交量及未售庫存仍有改善空間。不同 stock bundle 不能直接用單場 raw 判斷 persona 孰優或比 TA 弱多少；v1／v2 同條件對照尚缺後半段。

v2 規劃時每卡 free 約 23.5 GiB，啟動時約 13 GiB，低於設定需要的約 21 GiB；vLLM 明確回報 `Free memory on device ... less than desired GPU memory utilization`。v2 行動數 0，是啟動資源失敗，不能當作策略 0 分。當時 queue 只辨識預檢顯存不足，未辨識此啟動錯誤，因此 Seller 停止，controller 繼續完成獨立 Buyer 工作。

已在兩個 queue 增加僅限「本次新 wrapper／0 行動／精確啟動顯存錯誤」的重排判斷與證據 log；不把舊 log、一般 OOM 或已執行失敗吞掉。15 項 CPU 測試通過（`/tmp/adl_queue_startup_fix_cpu_tests.log`），含真實失敗 log 辨識與舊流程回歸；尚未以新 GPU 啟動實測，不修改凍結的評估來源、persona 或權重。

使用者在專案根目錄補跑以下指令；保留已完成 v1，只重播未完成 v2，Buyer 不重跑。顯存不足會排隊，有一卡先跑，兩卡合格則雙卡；等待時間未知。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/queue_seller_personas.py \
  --resume results/basic_checks/20261121-005859_seller_personas_1f4wxek7 \
  --gpus 0,1
```

跑完讀 Seller 資料夾的 `persona_comparison.md`。獨立核帳與來源指紋證據見 `docs/OVERNIGHT_RESULT_AUDIT.json`。
