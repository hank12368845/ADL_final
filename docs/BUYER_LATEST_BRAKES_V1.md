# Latest + step84 小規模煞車補課

2026-11-20更新：已完成run `20261120-024106_buyer_latest_brakes_vjsgyqu9`。
屬部分改善但未過煞車驗收，不直接重跑或續epoch；先讀該run的independent_brakes_review.md與Guideline第38.4節。

2026-11-20。依使用者最新指示，固定三份現用persona原文，從官方step84接續微調。
這輪只補現金、持有／當前邊際效用，以及判定拒絕後結束思考；不改評論優先序、不重寫persona。
Codex只做CPU準備，GPU訓練與推論由使用者啟動。

## 先確認原始錯誤

來源results/basic_checks/20261120-020252_step84_repair_ewp82xy6，
逐題觀測／原始文字在known5_cash_persona_audit.json，可讀表格在known5_cash_persona_audit.md。

| known失敗 | 實際persona | 現金 / 價格 | 判定 |
|---|---|---|---|
| board equal negative | p1 | 58.63 / 58.64 | 買不起仍buy_listed |
| offer above_one negative | p2 | 58.63 / 58.64 | 買不起仍accept_offer |
| offer zero negative | p1 | 0 / 58.64 | 零現金仍accept_offer |
| offer equal negative | p1 | 58.63 / 58.64 | 買不起仍accept_offer |
| offer wide negative | p3 | 29.32 / 58.64 | 已知不足，重複推理到1024tokens，無工具 |

known五題是4次買不起＋1次真實思考截斷，沒有重複購買。
五次現金錯誤合計：p1=3（全known），p2=2（known1＋official1），p3=0。
official是p2用6.60買10.00，private_state已寫不能買，工具仍buy_listed。
fresh另外2次p3重買：拿system初始估值當當前效用，忽略units_bought與缺少的next utility。
case名稱含歷史_p1不代表實際persona；以凍結metadata中的persona_index判定。

step84原SFT使用官方原始persona，不是短版。短版用於後來的診斷。
本輪全面用latest可直接處理這項輸入差異，但不能先宣稱差異已證實是退化主因。

## 固定輸入與監督

- 三份latest由agents/example_student/buyer讀入，原文6403／7600／7031字元，保存SHA；不加新規則。
- 每個訓練prompt都有完整latest；不把現金／價格另外算好貼入observation，不依賴改TA harness。
- 官方複習保留原始世界事實與目前觀測，只換persona與目標答案；超長只移除最舊歷史，仍可保留最多6turns。
  system＋當前觀測仍超12288則換另一個官方來源，不截掉財務資料。官方舊世界horizon／多件表的差異仍列為限制。
- 目標think為程式從可見資訊產生的短理由，不冒稱官方teacher原始思考；最多160tokens。
  順序為來源ID→已持有／當前效用→現金與總價整數分比較→必要時一次品質估算→直接工具。
- p1／p2現金對照包含等額、差一分、零現金、明顯不足；board與offer都有。
  前12步就涵蓋差一分與零現金，避免關鍵類型都排到最後。
- p3有已持有仍掛牌、初始價值很高但當前效用缺失的陷阱。先否決持有，不再引用初始價值。
  變品質正例按其初始q=2/3、low_multiplier=.4作一次E=.8V估算；定價低於低品質效用，成交也有正收益。
- private_state放第一參數，內容依當前事實生成；不先把尚未成交商品寫成已持有。
- 不新增tool-name輔助loss；仍用普通assistant completion CE，prompt／history mask。

## 資料與防止全wait

training/data/buyer_latest_brakes_v1：288筆完整一輪。

| 類別 | 數量 | 來源 |
|---|---:|---|
| 官方成功交易複習 | 144 | 原official train中24筆buy、24筆offer，各配3persona；保留原正收益交易 |
| 對比補課 | 144 | 12組official train來源商品／效用世界，現行引擎產生board／offer的現金、持有、負收益控制 |

補課引擎盤面是明示的counterfactual，不是原官方完整軌跡：使用原商品／首件效用，
改為現行單件效用與5×10市場；fixture成本等私有生成設定只用於引擎，沒有當作買家輸入或學習目標。
正例由原引擎實際成交確認正surplus；負例實際嘗試確認insufficient_cash、拒絕或非正收益。
不把本次known／fresh／official失敗原題回灌訓練，官方封存test不讀取推論。

每個global batch8固定3 buy_listed＋3 accept_offer＋2 wait；同經濟條件的board／offer正反完整保留。
總動作108 buy／108 accept／72 wait，交易75%、wait25%，每筆一次、288筆全覆蓋。
每卡144筆曝光，避免早期wait標籤偏斜。這是小型操作補課，不宣稱最優市場策略。

## 訓練配方

從training/runs/20261119-213016_buyer_official_u3csnqyh/checkpoints/step_0084載入權重，
同底座revision、rank16／alpha32、語言層，bf16、gradient checkpointing。
新AdamW、LR5e-6、warmup3、cosine至36steps；global batch8、兩卡各micro1×accum4。
保存本輪step12／24／36及完整optimizer／scheduler／rank RNG，原step84保留。
這是低LR新補課，不假稱恢復原step84 optimizer。中斷後resume則恢復本輪完整狀態，不重開warmup。

## 訓練中直接驗證

每次67個native HF案例：已測latest回歸37、不同官方validation來源群組的新控制24、實際seen6。
初始step0重用37筆已核對的父模型輸出，另外生成新24＋seen6；三個checkpoint各生成67筆。
全程約231次新推論，兩卡固定同題與pair路由。seen來自首12步真正batch prefix，step12起確實看過。
新validation與訓練來源group分離；它是development，不當作封存test或正式分數。
使用1024newtokens、greedy、正確EOS；正式市場仍有官方temperature與每round token預算。

選模必須保住原known8/8與fresh4/6的捕捉，各自不能以另一組改善抵銷退化；
新題至少10/12且不比父模型低超過1題，現金／重買／無工具／截斷均0，官方snapshot合法率不退。
檢查已見題以區分尚未學會與泛化失敗。低loss不替代自由生成。
正例捕捉大幅退化會停損；無合格候選就明寫None，不自動部署、接第二epoch或啟動市場。

## 使用者執行

資料已準備，直接：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_latest_brakes.py --gpus 0,1
```

同一命令包含訓練前與每12步評估。依前輪速度先預留約50～80分鐘，非實測保證；以實際progress／ETA為準。
開頭載入／最長樣本前後向預檢顯示估算中，確認training mode及checkpointing；不再重現先前eval預檢錯誤。
需要兩卡各至少18GiB啟動預算（已扣餘裕），不停止他人工作；資源不足直接說明，不空等。

輸出training/runs/<timestamp>_buyer_latest_brakes_*，先讀training_review.md；
native_step_*.json有逐題輸出，checkpoint_scores.json有門檻，finished.json有候選／停損狀態。
runner.log、metrics_rank*.jsonl、samples_rank*.jsonl、hardware_rank*.json與batch_plan.json保存實際執行證據。
personas/是本輪原文，agent.yaml為reasoning:true，不覆寫提交agent。

CPU檢查：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_latest_brakes.py --cpu-check
/home/hank/.cache/adl_final_venv/bin/python -m unittest training.test_buyer_latest_brakes
```

中斷後接續（僅完整本輪step12或24）：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_latest_brakes.py \
  --resume 實際run/checkpoints/step_0012 --gpus 0,1
```

資料／persona／來源程式或配方不同會拒絕exact resume。重現資料使用buyer_latest_brakes_data.py，拒絕覆蓋既有輸出。
方法由Codex協助，需在report揭露LLM使用；實際改善與硬體時間待使用者GPU執行確認。
