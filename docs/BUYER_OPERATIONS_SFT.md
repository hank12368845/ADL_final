# Buyer操作LoRA第一階段（2026-11-18）

目標先消除可避免的無效／零效用動作，再比較高分策略。程式、設定與完整重現說明見
[training/README.md](../training/README.md)。GPU由使用者啟動，尚無訓練或收益改善結果。

## 轉換方法的證據

`results/basic_checks/20261118-170543_buyer_diagnostic_0yk6z83o`：144calls全部取得模型回應。
v3/v4各72次，action通過48→39、完整state通過23→22、零邊際效用購買錯誤10→20。
保留歷史時action為16/36→11/36，去歷史32/36→28/36，顯示歷史干擾值得優先處理。
只有6個開發案例及2個seed，且v4同時改多項規則，不能證明prompt長度是唯一原因或模型已達能力上限。

## 現在準備好的內容

- 可重現的合成教具：80個市場group，先切64/8/8，再展開10類案例與3份persona。
  train1920／validation240／test240，無超長樣本；可含0～6回合歷史。
- 官方工具模板、thinking OFF，只監督最後正確回答。舊錯誤只作輸入，不會一起被當成答案訓練。
- 原始engine逐筆驗證目標合法及成交正收益；教師只看公開給此buyer的資訊。
- 短記憶只存bad／pending／last_buy，held／cash直接讀當前observation。
- rank16、alpha32、LR5e-5、80steps、global batch8。預設兩卡DDP；每卡microbatch1、累積4次。
- checkpoint為候選，不自動安裝至active agents；保留step40／80，避免只靠最終loss選模。

未知品質暫採low倍率的保守下界，且暫不教有利的多件購買或low→high升級。
這是為了先隔離操作問題的訓練政策，不能認定為最優策略，之後仍以實際surplus決定是否放寬。
沒有下載或混入TA checkpoint、既有真實失敗題或其變形。

## 使用者現在執行

CPU資料與檢查已由Codex完成，直接啟動8步價值驗證版即可（80步recipe保留供確認值得後使用）：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py
```

程式依當下剩餘顯存挑最多兩張卡，各留2GiB，訓練預算不足18GiB的卡跳過。
明確單卡加`--gpus 0`；指定兩卡加`--gpus 0,1`。
如要先驗證GPU管線可加`--max-steps 8`，但八步不代表充分訓練。

啟動包含模型載入、最長序列前後向檢查、初始validation、8訓練steps及兩次validation。
8步probe使用12筆validation，涵蓋全部10類任務，step4／8即保存；80步recipe需明確加
`--config training/configs/buyer_operations_v1.yaml`。已啟動的舊run不會套用新設定。
各階段有狀態／進度；未觀測到速度前ETA顯示估算中。Torch線性注意力fallback目前未加速，
不能預先保證訓練時間。前後向記憶檢查尚不含第一步才配置的Adam狀態，仍可能OOM。

輸出是新建的`training/runs/<time>_buyer_operations_*/`，包含runner.log、metrics.jsonl、
train_config.json、hardware_rank*.json、checkpoints與final_adapter。
Ctrl+C停止本次工作、保留已完成checkpoint；目前不支援optimizer完整resume。
使用者跑完提供run路徑，再分析loss、實際資源及行為驗證，不由Codex空等GPU。

## 怎麼判斷值得繼續

先用相同短persona及相同held-out題，配對比較plain base與candidate：

```bash
PYTHON_BIN=/home/hank/.cache/adl_final_venv/bin/python \
  bash training/evaluate_buyer_operations.sh training/runs/<本次run>
```

10題×3persona×2seed×2模型=120calls。看action、最小state、policy以及operation capture。
Operation capture只在有可執行保守正收益交易的題目計算；全wait不能混成成功。
新state rubric為diagnostic_operations_v1，與舊完整檢查表的state百分比不直接比較。

訓練loss下降、CPU fake-client測試成功、或模型在訓練題輸出正確，都不代表得分提高。
未見過的合成題改善後，再跑既有真實失敗輸入和完整30turn市場；最後仍要不同seed、
商品／對手及輪換下的surplus證據。Seller訓練與最終agent選定尚未進行。


## 先取得是否值得的證據

使用者明確說明可以接受長時間LoRA，但應先確認值得。原80步首個checkpoint在40步，
讓初次價值驗證太晚。已新增8步probe，並在base/candidate配對完成後自動產生operation_gate.md/json，
逐persona列出無效動作、政策失敗、交易機會捕捉及配對修正／退化數。
指標改善只是此小測的證據，不是統計顯著或市場高分保證；無改善也不證明LoRA學不會。
長版仍保留，獲得正面證據後才投入。現在尚不支援完整resume，長版會從底座重新訓練；
不能把新啟動80步稱為保留optimizer的續訓。
