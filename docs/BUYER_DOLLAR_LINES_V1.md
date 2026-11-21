# 美元一行檢核：先6步停損，再驗未見題

2026-11-20，使用者同意此安排並要求直接可執行指令。GPU由使用者啟動，Codex只完成CPU準備。

## 本輪改動

起點為官方step84，Thinking ON，rank16／alpha32。現用三persona、system／observation／工具／歷史原樣保留。
只重新標記現有48個train控制盤面的assistant草稿，沒有新世界、商品、觀測變體或策略；正式harness不改。
草稿直接使用觀測的美元單位，示例：

```text
CASH=19.59 | ROW=offer:o1:P74 total=19.60 held=0 val=58.81 adj=58.81 afford=0 eligible=0
```

一行只描述本次檢查的候選，減少先前「列全看板／巢狀JSON」的學習負擔。控制盤面已驗證正側僅此候選符合資格、負側無其他合格交易。
這不是在正式市場規定只看一件商品；更廣候選選擇能力仍待市場驗證，不能把小集通過當高分。
board的ROW包含seller與product；offer使用當前唯一offer ID與product。
`val`讀當前下一件效用，持有者缺失即0；p3控制題`adj`按原先可見品質先驗調整，不讀隱藏grade。
`afford`僅比較當前現金與總價，`eligible`還需未持有、live／qty1及正收益。

parser用精確Decimal比較美元，內部以最小貨幣單位做比較不等於要求模型換算；19.59寫成1959仍錯。
19.6與19.60相等；19.601不四捨五入偷給分。標點／空白容錯與逐欄扣分保留，不從工具private_state回填草稿。

工具目標與private_state沿用原來已驗證的答案，CPU逐題核對一致性。private_state仍是工具要求的JSON字串，只有think草稿改成文字。
persona要求wait時pick／交易flags為null，故「檢核候選afford=false＋wait的private_state無交易null」不算矛盾。
評估另列private_state一致性，不將自訂草稿格式／state檢查冒稱引擎違規。

## 訓練監督

`loss = 完整檢核區平均CE + 七欄答案值平均CE + .25 × 原生工具回應區平均CE`。
完整檢核區包含第一個CASH token、來源／欄位名稱、數字、判定，以及原生結束think標記；工具／private_state不能稀釋這段分母。
不是只提高第一個token或工具名稱權重。prompt／history不算loss；每段及每欄loss分開記錄，仍以自由生成驗收為準。
檢核區最長57tokens（含結束think）；全序列平均約8074、最長10030tokens，長system／當前觀測未裁剪。

48個盤面：足額／差一分、足額／零現金、未持有／持有、正收益／非正收益，三persona×兩通道均涵蓋。
仍是原先各組正反pair，不宣稱同商品同盤面已有三種現金版本；沒有為湊三連題新增資料。
每batch8為2buy／2accept／4wait、四組完整pair；首6步每題恰見一次，後6步再完整一次。
新AdamW、LR1e-5、warmup2、cosine到12，最多12步。warmup也有更新。

## 自動階段與停損

父step84的73筆自由生成直接從前輪原始紀錄重算並明標cached，不再花GPU重跑父模型。
驗證從原生空白think開始，沒有CASH前綴、格式答案或數字提示。

### 第6步

先保存adapter＋optimizer／兩rank RNG，再只推論原來的seen12（每卡6）。
門檻全部預先固定：

- 12/12自主以CASH開頭、12/12候選七欄正確；不靠寬鬆parser放過錯數字或ID。
- 保住6/6正例捕捉。
- 三個差一分負例全部安全、三個持有負例全部安全，現金／重買／無工具／截斷零次。
- private_state當前事實與交易／不交易flags正確，檢核與最後動作一致。

任一未過：`status=stopped_step6_gate`，正常exit0，立刻結束；不訓練step7、不推論未見24、不啟動其他工作。
這是本配方的成本停損，不代表模型或一行方法永遠學不會。seen12沒有零現金題，不能聲稱step6涵蓋全部條件。

### 第12步（第6步通過才會執行）

持續相同optimizer／scheduler，最多再6步；保存並自由生成全部73題：seen12、未見來源24、原題37。
未見來源24含p2零現金與三角色持有對比，屬已使用過的development validation，不是封存test。
逐欄、persona／通道分項驗收，負例安全、原known／fresh捕捉分別保住、官方snapshot不退，並看state／動作一致與截斷。
保留父模型對照，不用全wait換安全，也不靠兩題組互換掩蓋退步。
通過也不自動擴資料、升級提交adapter或市場；第三階段另看結果再安排。

## 指令與輸出

資料已CPU備妥，一條命令包含訓練與分階段驗收：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_dollar_lines.py --gpus 0,1
```

首關先預留約8～15分鐘；若通過並完成第二關，整輪約15～30分鐘。這是依前輪速度估算，新配方尚無GPU實測，以現場ETA為準。
兩卡各自扣除顯存餘裕後需≥18GiB；啟動前查磁碟與cache，不終止他人工作。載入／preflight標估算中，訓練與推論顯示完成量、耗時、速度、ETA。
輸出`training/runs/<timestamp>_buyer_dollar_lines_*`，先讀`dollar_review.md`。
`finished.json`會區分正常完成與step6停損；`runner.log`是終端全文；逐題原文與欄位證據在`native_step_*.json`。
`samples_rank*.jsonl`保存實際曝光，`metrics_rank*.jsonl`保存獨立各段／各欄loss與GPU記憶體。
原step84、先前JSON實驗與提交persona／adapter均保留；不提供隱式續訓或自動重跑。

CPU重現檢查：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_dollar_lines.py --cpu-check
/home/hank/.cache/adl_final_venv/bin/python -m unittest training.test_buyer_dollar_lines -v
```

重建資料需先有舊版資料與官方基線，使用`python -m training.buyer_dollar_data`；既有輸出拒絕覆寫。
CPU測試不執行真實HF模型，涵蓋美元／欄位獨立評分、原始輸入／persona／pair與曝光、原生模板／工具parser、
完整檢核首token與關閉think的loss遮罩、CPU小logits梯度，以及第6步門檻與mock雙卡launcher。
