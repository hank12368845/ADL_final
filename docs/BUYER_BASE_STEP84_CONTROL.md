# 原persona、裸底座 vs step84：固定36題模型對照

2026-11-20。使用者已採納先停止格式／persona小修，確認LoRA效果，再做一場正式尺寸市場診斷。
本輪只準備推論，不訓練、不啟動市場、不修改agents或adapter。GPU由使用者啟動。

## 唯一改變：是否使用LoRA

- 保留 `agents/example_student/buyer/persona_1.md`～`persona_3.md` 的原文。
- 原始observation、情境、歷史、工具、題目、正負pair、逐題GPU rank均不變。
- Qwen/Qwen3.5-4B，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- Thinking ON、greedy、不抽樣、bf16／SDPA、max_new_tokens=1024、原生chat template及EOS `[248044,248046]`。
- 12個seen＋24個已使用過的development；每persona 12題，正例18、負例18，非封存test或市場成績。
- step84原36筆輸出重用並重新評分；只新增裸底座36次推論，每卡18次，兩個獨立副本。
- 裸底座直接AutoModelForCausalLM載入，不建立PEFT、不載入任何adapter；程式驗證模型沒有LoRA參數或peft_config。
- 推論plan移除teacher answer／label_action；模型只收到原messages與tools，不提供真實數字答案或資格提示。

固定reference是 `results/basic_checks/20261120-150124_compact_buyer_g5p78gcy/plan.json`。
使用其中**original** prompt SHA與cached原輸出，不用失敗的compact prompt。
啟動前核對reference、原資料、原persona、step84權重、來源程式與套件版本；任何差異中止。
step84本輪不重跑，因此這是固定cached輸出的控制，非對全部硬體／數值誤差重新配對的重複實驗。
相同GPU rank沿用原路由；實際卡與硬體名稱另存。診斷不是正式vLLM抽樣市場驗證。

## 使用者執行

在repo根目錄直接貼上：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_base_step84.py --gpus 0,1
```

不需要填NEW_RUN或前一個訓練目錄。預設使用固定的已完成reference。
兩卡須各有18GiB啟動預算，先扣2GiB餘裕；不足就中止，不終止別人的工作、不悄悄換路由。
按逐卡剩餘顯存設定PyTorch memory fraction。結果空間需2GiB、runtime cache需5GiB。
空間不足按既有規則明示fallback到/tmp；結果持久保存仍需之後移回有空間磁碟。
上一輪同規模36calls約4分15秒，可先預留3～6分鐘，實際以終端ETA為準，並非時間保證。
載入／首題warm-up顯示ETA estimating；每10秒整體心跳及逐卡18題進度、elapsed、速度、ETA寫入runner.log。

輸出 `results/basic_checks/<timestamp>_base_step84_*`，或明示的/tmp fallback：

- `base_step84_review.md/json`：總體、seen／development、各persona、board／offer分項與差值。
- `decisions_rank0.jsonl`／`decisions_rank1.jsonl`：36次真實新推論、原文、prompt SHA、實際token數、沒有adapter的標記。
- `rescored_decisions.json`：完整36題用相同rubric重算。
- `semantic_review.json`：共72份四要素語意證據，初始pending；結果回來由Codex逐題讀，不能用關鍵字冒充正確率。
- `plan.json`：資料／模型／來源SHA、版本、路由、顯存與cache規劃；`personas/`保存原文。
- `hardware_rank*.json`、`runner.log`、`wrapper_result.json`：硬體、實際路由與執行狀態。

## 驗收與後續

主要看捕捉正收益與避開負例能否同時保住；另外看同一pair「正例捕捉＋負例安全」的both_correct。
所有wait可能合法率很高，但正例捕捉為0，不當作成功。現金錯誤降低卻改成重買／無工具，也不當修好。
Valid同時包含工具與經濟約束，不是單純engine拒收率；自訂private_state一致性另外列，不當引擎違規。
報表不硬合成總分、不自動選勝者；逐persona的退步與修好題單獨列出。

若底座較好，審查官方資料的舊世界效用／行為遷移，不預設繼續修step84；不因一次診斷即證明SFT有害。
若step84更好，保留它的有效改善並按市場損失補強；若各有取捨，明寫不確定，不硬選。
結果讀完再準備一場5seller×10buyer、5round×6turn市場基準，量surplus、無效回合、零邊際效用支出與品質損失。
本指令結束後**不自動訓練、不部署、不自動開市場**。

CPU檢查（Codex執行，無GPU）：

```bash
/home/hank/.cache/adl_final_venv/bin/python -m unittest training.test_base_step84 -v
```

`--prepare-only`只驗證與保存計畫，不查GPU、不啟動模型。`--report <實際結果目錄>`可CPU重新產生報表。
CPU fake輸出僅驗證程式，不是裸底座表現，不得放入正式模型結果。

CPU驗證已完成：5項測試通過（34.57秒）；加入進度條後重驗進度測試通過（9.41秒，包含現行plan重建）。
證據見docs/buyer_base_step84_cpu_review.json。Codex未執行任何新GPU推論／訓練。
