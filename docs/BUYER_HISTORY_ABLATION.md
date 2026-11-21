# Buyer 第一步：checkpoint／歷史／記憶對照

目的：在再次訓練前，確認全部 `wait` 的現象與早期 checkpoint、歷史對話、
`private_state` 分別有何關聯。本步沒有訓練更新，只有模型推論；不改正式 engine、
現用三份 buyer persona 或既有 checkpoint，也不使用模型外工具修復。

## 執行

Codex 已準備程式與 CPU 檢查。GPU 工作由使用者在專案根目錄啟動：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_history_ablation.py --gpus 0,1
```

預設使用已完成 run `training/runs/20261118-193743_buyer_operations_6yy413cv` 的
`checkpoints/step_0004` 與 `checkpoints/step_0008`，加上 plain base。
啟動前查磁碟、快取、兩卡可用顯存；依餘裕設定各卡預算。
明確指定兩卡但不足時直接停止，不默默改單卡，也不終止他人工作。
如需避開既有服務，可追加 `--port 8100`（使用 8100／8101）。

終端顯示完成數／144、耗時、速度、ETA；模型載入與 warm-up 尚無速度時顯示 estimating，
並顯示服務 log 最新狀態。新結果預設保存於 `results/basic_checks/<time>_buyer_diagnostic_*/`；
磁碟不足時可能改用 `/tmp/adl_final_basic_checks/`，啟動時會印出實際路徑。
先讀 `history_ablation.md`；原始回答在 `decisions.jsonl`，
其他證據含 `history_ablation.json`、`scored_decisions.jsonl`、`plan.json`、
`job_routes.json`、`runner.log`、`vllm/`。不要把 prepare-only 資料夾當成已完成推論。

## 為何是 144 次

固定短版 `operations_p1` persona、thinking OFF，其餘 decoding 沿用本地正式模型設定。

| 維度 | 條件 |
|---|---|
| 情境 | 2 個 held-out validation 市場 × 4 題 = 8 題 |
| 模型 | plain base、step4、step8 |
| 上下文 | recorded、fresh、clean_fresh |
| 推論 seed | 0、1 |
| 總量 | 8 × 3 × 3 × 2 = 144 |

選用 validation 的 group `20337802`、`20302487`，不重用前次單一 test group 的十題，
並保留未使用的 test 市場供後續驗證。這兩個 validation 市場從此屬於開發診斷資料，
不是未見過的最終測試證據。

每個市場四題均使用固定品質商品，避免清除品質記憶改變合理交易風險：

| 題目 | 應做動作 | 驗證目的 |
|---|---|---|
| 正收益掛牌 | buy_listed | 讀取當前配對、現金、效用，抓住可交易機會 |
| 同掛牌、現金差一分錢 | wait | 不因低價就買，須核對可負擔性 |
| 正收益 binding offer | accept_offer | 能接受有利私人報價 |
| 同 offer、報價超過效用一分錢 | wait | 合法可買但虧錢時應拒絕 |

正例精確重現來源資料當前 observation，負例以受控修改建立；所有目標動作由原引擎核對。
現金負例的嘗試購買必須得到 `insufficient_cash`；高價 offer 雖合法成交卻損失一分錢。
這些是有引擎 backing state 的合成診斷，不聲稱是模型完整市場軌跡。
答案與檢查結果不放入模型 prompt。

## 如何分開測歷史與記憶

| 名稱 | 歷史對話 | 當前 private_state | 當前 observation |
|---|---|---|---|
| recorded | 保留 | 保留 | 保留 |
| fresh | 移除 | 保留 | 保留 |
| clean_fresh | 移除 | 清空 | 保留 |

`recorded → fresh` 比較歷史對話影響；`fresh → clean_fresh` 比較備忘錄影響。
清記憶只改 `your_private_state`，不清除當前成交結果、訊息、報價、現金、持有量等資訊。
board 題有故意錯誤的舊 held／cash 備忘錄，offer 題原本備忘錄為空。
因此記憶效果只用非空備忘錄的 8 對回答；空備忘錄的另 8 對是相同輸入重複控制。
若同 seed／GPU 的相同輸入仍有動作差異，需考慮服務批次或生成波動。

所有模型／上下文的相同題目與 seed 固定同一 GPU；兩張卡各 72 calls。
工作佇列交錯送到兩卡，避免前四個並行請求全擠到同卡。
`job_routes.json` 是計畫路由，原始回答中的 `gpu` 是完成工作路由；報表核對配對是否一致。
這是獨立副本分工，並非 tensor parallel；總 calls 相等也不保證每秒利用率完全相同。

## 怎麼讀結果

每個模型／上下文有 16 次：8 個正例、8 個負例。

- 主要看正收益捕捉 `captured / 8`，另外拆掛牌與 offer 各 `/4`。
- 同時看負例是否正確等待，以及交易 precision；零交易時 precision 為 N/A，不能當 100%。
- 動作正確與記憶錯誤分開列；全部 wait 即使合規，也只能答對負例。
- `paired_checkpoint_effects` 顯示 base→step4→step8 的修復與退化；
  `paired_context_effects` 顯示相同題目／seed 改上下文後的修復與退化。

若 step4 有買、step8 沒買，支持偏移在更新過程中加重；若移除歷史後恢復，
支持歷史敏感性；若僅清除備忘錄後恢復，支持錯誤備忘錄敏感性。
這些結果提供方向，不能單憑小樣本宣稱某一因素是唯一原因。
若 step8 在所有條件均不交易，仍須檢查訓練與 vLLM 的 forward／動作分數一致性，
再把問題完全歸於權重。本步沒有計算 240 筆 decision token log-probability，
也沒有調 sampler、資料比例或開始新一輪 SFT。

本步不是市場 surplus、TA-normalized grade 或高分證明。它是選擇下一個修正方向的低成本診斷。

## CPU 驗證

```bash
/home/hank/.cache/adl_final_venv/bin/python -m unittest \
  training.test_buyer_history_ablation training.test_chat_payload training.test_buyer_operations -v
```

涵蓋原引擎正負例、split 邊界、三種上下文差異、實際 vLLM HTTP request schema、
完整 144 次 fake-client 雙卡路由與報表、缺失／錯配輸入拒收，以及原有 HTTP400 防護。
fake-client 的模擬恢復數字只驗證程式，不代表真模型會恢復。
`--prepare-only` 可產生凍結輸入與 plan，不查 GPU、不開服務。
