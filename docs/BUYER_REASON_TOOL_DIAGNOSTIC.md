# 拒絕理由與工具選擇的小型診斷

2026-11-20。停止追加訓練，先定位原step84與補課step36的現金／持有拒絕問題。
本輪由Codex完成CPU分析與準備；GPU推論由使用者啟動，沒有新訓練。

現有可見輸出已提供線索：

- 已見p1題現金19.59、價格19.60，step36只比較價值58.81與價格，然後buy_listed，未生成現金不足的拒絕理由。
- 已見p2相同現金反例也是直接accept_offer；private_state偶爾抄成cash0但仍afford=true，抄數字不等於形成拒絕。
- 重買題把system初始估值當當前next utility，雖然state列了held，選擇仍當成unowned。

以上是生成文字證據，不是內部認知／算術能力的完整證明；也還不能排除理由到工具的連結同時有問題。

## 固定對照

12題：三persona的已見cash正反6題，與三persona的新validation owned-board正反6題。
兩個模型：原官方step84、補課step36。所有case保持原latest persona、觀測與工具。

每個模型兩個條件：

1. free_cached：重用上一輪同模型、同輸入、同GPU的既有自由生成，共24筆，明確標記來源SHA。
2. fact_prefix：原generation prompt後提供正確的事實算式並關閉think，再讓模型自行生成工具；共24次新GPU推論。

提供的reason列出current cash、完整held集合、每個board／offer的當前next utility、總價、
cash-total、cash>=total、品質估值與eligible真偽；不含wait／buy_listed／accept_offer或tool-call答案。
p3的變品質項來自本輪受控世界，無既有variable-grade紀錄，按其persona的初始q2/3、E.8V估算。
包括所有當前候選與持有陷阱，不只提供一個看似能買的商品。

權重、原輸入、greedy、1024newtokens、EOS與同題GPU固定；兩卡各12次新推論。
追加正確算式是**診斷介入**，不是正式允許的observation formatter，也不是模型自己推理成功。
提供前綴會改變分布／文字長度，不能把結果當作唯一因果證明。

## 看什麼

同時報6個正例捕捉、6個負例安全、現金／持有錯誤、無工具與截斷，保存每題原始續寫。

- 若提供正確事實後能穩定拒絕、也保住正例：支持優先處理事實抽取／約束理由生成。
- 若正確事實已在前綴，工具仍選買不起／已持有者：顯示工具選擇也有弱點，不能只增加算式示範。
- 若只會等：正例捕捉會顯示失敗，不能當煞車修好。
- 若仍截斷／重開思考：先處理輸出收束，不把無工具直接解讀成買等偏好。

這些是小樣本development診斷，不是新checkpoint、未見正式成績或市場surplus。不自動開啟下一輪訓練。

## 使用者執行

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_buyer_brakes.py --gpus 0,1
```

24次推論，先預留約3～8分鐘，以實際progress／ETA為準。載入時顯示估算中；雙卡逐次及整體進度、速度、elapsed、ETA與log皆保留。
輸出results/basic_checks/<timestamp>_brakes_reason_tool_*，先讀reason_tool_review.md／json。
plan.json含原題、提供的算式、權重／程式／既有結果SHA；decisions_rank*.jsonl含原始工具續寫與錯誤。

CPU準備：同指令加--prepare-only；CPU回歸：python -m unittest training.test_diagnose_buyer_brakes。
