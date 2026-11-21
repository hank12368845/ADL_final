# ID repair v1：當前賣家與 offer ID 的來源

從上一輪board repair step32做新一輪小幅SFT，針對3次歷史賣家混配、2次通知／虛構offer錯誤。
資料從training市場重新生成，沒有把那5個validation題放進訓練；仍是程式示範SFT，不是模型自生成偏好資料。

## 訓練指令

CPU資料已備好，使用者執行：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_id_repair_v1.yaml --gpus 0,1
```

parent是training/runs/20261119-134251_buyer_operations_50lhkufb/checkpoints/step_0032。
載入LoRA、新建AdamW／scheduler，LR5e-6、warmup4、本輪32steps，保存本輪step16／32。
thinking ON、r16／alpha32、batch8；雙卡各128筆，原權重保留，不自動部署。
按上輪32分55秒粗估30～40分鐘，以終端進度／速度／elapsed／ETA為準。
啟動前檢查磁碟與逐卡顯存，兩卡不足即停止；輸出新training/runs目錄與runner.log、metrics、source/data hash及sample trace。

## 這輪的256筆資料

- 128個board正例、64個board負例：16個training市場，各有fresh／recorded，
  保持同一段歷史、商品、價格與現金，對照當前seller A／B，正確工具必須跟著當前掛牌改變。
  一半使用歷史賣家目前真的有掛牌的情況，避免學成永遠拒絕舊賣家。
- 每個board狀態都有原引擎真正產生的成交通知，offers_to_you為空。
  每例都用engine驗證：錯賣家不能買、通知ID不能接受；正例合法成交且surplus>0。
  負例涵蓋現金不足、已持有、缺貨與負surplus；各自驗證引擎拒絕或成交會虧損。
- 32個offer正例與32個負例：保留原reasoning-v1的輸入及正確答案，排除前兩輪見過的source rows，作行為保留。

board答案先複製當前seller／product／price／stock，明寫來源是current board、offers_to_you_count=0，
再核對同商品cash／held／utility，最後重複同一tuple的工具。
沒有把錯誤ID寫死成m5、offer_0或任何validation賣家，靠當前欄位的有效成員決定。
觀測由原引擎產生，使用原始JSON格式與原persona，TA引擎沒有修改。
validation／test與64題cash probe逐byte保留。樣本ID有256個，部分不可見stock對照可能產生相同輸入，
不將這些重複條件稱為獨立市場證據。

## 跑完評估

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_board_repair.py \
  NEW_RUN --gpus 0,1
```

NEW_RUN換成這次訓練印出的新run目錄；同時比較這次parent、本輪step16、本輪step32。
仍為1152 calls，雙卡各576，有服務載入與呼叫完成進度、速度及ETA。
先讀輸出資料夾的board_repair_review.md，再看完整cash_decision_pairs.json與reasoning。
目標保住board至少72/96、offer正例88/96與pair78/96，並把board錯ID／假offer及現金錯誤降為0。
本地目標不等於正式grade，仍有其他正例wait，也需要完整市場surplus評估。

CPU檢查包含engine-backed反例、只改當前seller的對照、歷史／新賣家正例、資料隔離、
offer原答案保留、原生thinking mask、雙rank取樣與三模型prepare-only入口；未執行真模型GPU。
證據docs/buyer_id_repair_cpu_review.json。新機器先重現reasoning-v1及board-repair-v1，
用id_repair_data.py --parent YOUR_PARENT重新準備，再將本版config的initial_adapter設為同一路徑。
