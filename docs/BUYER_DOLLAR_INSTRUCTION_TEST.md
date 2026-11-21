# 明確美元檢核指示：step84與step6純推論對照

2026-11-20使用者已明確要求執行此方向。GPU仍由使用者啟動，不再訓練。

## 本次條件

- 權重固定：官方step84、本輪美元補課step6（20261120-141718_buyer_dollar_lines_sgkzzv1q）。
- 同一組seen12：三persona、六正六負、三個差一分負例與三個持有負例。原兩模型共24筆自由生成作cached對照。
- 新增條件：只在診斷persona副本末尾加入329字元的美元格式／比較規則，內容固定於docs/BUYER_DOLLAR_FORMAT_DRAFT.md與腳本常數。
- 不提供實際現金、價格、商品／offer ID、運算結果或eligible答案；原觀測／工具／歷史／情境都不改。
- 不在think預填CASH或JSON開頭。模型從原生空白think自行生成整段檢核及工具；也不代它關閉think。
- Thinking ON、greedy、1024 new tokens、原生EOS與工具parser。兩卡各12次，同題／pair沿用原GPU。

新推論24次，零梯度更新。診斷persona長度6734／7931／7362，均≤8000；原agents persona與adapter保留。
條件明標format_instruction，不能把persona改動後的結果冒稱原始latest表現，也不能把seen小集結果當市場高分。
此指示包含一般格式與afford／eligible規則，不能將改善唯一歸因於第一個token或某項規則。

## 預先驗收

逐欄看cash、total、held、當前next utility、adj、afford、eligible；再看self-start、全欄正確、最後動作與state一致性。
工具合法獨立使用原snapshot評分，格式／state自訂檢查不冒充engine違規。
保留上一輪seen12門檻：12/12自主檢核與全欄正確、6/6正例捕捉、差一分3/3安全、持有3/3安全、無工具／截斷／cash／重買零次、state／動作一致。
不能因格式更寬鬆而靜默放寬現金／持有判斷。

判讀：

1. 父step84加指示也通過：優先擴大驗證persona變更，不急著再SFT。只有12個seen，不能直接提交或稱已泛化。
2. 只有step6通過：進一步驗證step6＋該persona組合，而非直接升級提交adapter。
3. 格式恢復而afford／held等仍錯：輸出形式已改善，條件判斷仍是待修重點；不宣稱只改格式就完成。
4. 仍不寫檢核：不能據此隔離條件判斷能力，先審查指示遵從，不直接重跑／長訓。

## 直接執行

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_dollar_instruction.py --gpus 0,1
```

依上一個24calls對照約2分鐘，先預留2～5分鐘，以終端實際進度／ETA為準。啟動前查磁碟、cache與逐卡顯存餘裕，不終止他人工作。
載入時顯示估算中，兩卡各12與整體24的完成量／耗時／速度／ETA保存於runner.log。
輸出results/basic_checks/<timestamp>_dollar_instruction_*，空間不足明示/tmp fallback。
先讀instruction_review.md／json；逐題raw在decisions_rank*.jsonl；plan.json保存原prompt與新prompt SHA、實際指示及資料來源。
personas/original與personas/format_instruction保存可比較的兩套原文。不自動寫回agents、不自動訓練或啟動下一個GPU工作。

CPU準備與測試：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_dollar_instruction.py --prepare-only
/home/hank/.cache/adl_final_venv/bin/python -m unittest training.test_dollar_instruction -v
```

CPU測試驗證觀測／工具不變、只有persona副本改動、指示不含實際答案、空白原生think、同題GPU路由、原生parser與逐欄評分、mock雙卡launcher。
CPU成功不代表模型遵從或GPU推論結果。
