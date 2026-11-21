# 先診斷檢核入口，再決定一行文字訓練

2026-11-20：使用者支持改用固定開頭、簡短標記的純文字檢核，也同意先用空JSON開頭對照定位。
本次不重訓、不拉長步數、不改persona／observation／部署adapter。

## 本次實際執行的對照

`training/diagnose_self_check_entry.py`固定比較官方step84與`20261120-130432_buyer_self_checks_rueclguj`本輪step12。
沿用這場的12筆seen：三persona、board／offer、現金／持有對比，六正六負，既有自由生成24筆明標cached。
新條件只在原生`<think>\n`後接`{"cash":`，沒有數字、商品／offer ID、持有狀態、答案、eligible或工具指令，也不代模型關閉think。
模型自行續寫剩餘草稿與原生工具，greedy、正確EOS、1024 new tokens，兩卡各12次新推論，同題／pair與原評估同卡。
合計24次新GPU推論，零訓練更新。原始輸入／資料／訓練程式SHA、adapter fingerprint及native prompt／prefill SHA均鎖定。

這是刻意介入入口的診斷，不是正式可提交的formatter或自主格式學習證據。
測的是「給空格式後是否能自己填對」，不能把供給的`{"cash":`算作模型自行學會起頭。

## 報表先看什麼

1. 是否開始寫現金數字、是否填對美分；即使後續JSON壞掉，第一個cash仍獨立量測。
2. 美元值正確但美分錯誤另外計數，例如觀測19.59、填19.59，不能偷偷乘100後給美分正確分。
3. 完整JSON、來源coverage、各欄（cash／total／held／next value／adjusted／afford／eligible）及整份正確率。
4. 正例捕捉、負例安全、cash／重買錯誤、工具與截斷；分persona結果保存在JSON。

若格式與欄位一起恢復，支持入口是瓶頸之一；若格式恢復但數值／資格仍錯，就不能只強化開頭。
若只第一個cash改善、後面格式仍壞，只能判定第一欄，不將其餘欄位缺失當成全部事實都錯。
若只有工具改善、沒有可核對的檢核，不能宣稱模型已學會完整檢核。
若父模型也恢復，不能把所有效果歸功於本輪12步訓練。
12筆seen只作診斷，不代表未見市場泛化或surplus。

由使用者啟動：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_self_check_entry.py --gpus 0,1
```

先預留約3～8分鐘，依終端實際進度／ETA；載入時標估算中。兩卡各12、整體24的完成量、速度、耗時與ETA均顯示並保存runner.log。
磁碟／cache／每卡剩餘顯存預算先檢查，不終止他人程序。
輸出`results/basic_checks/<timestamp>_self_check_entry_*/entry_review.md`；空間不足fallback會顯示暫存位置。
`plan.json`保留條件與原輸入，`decisions_rank*.jsonl`分開保存supplied_prefix、continuation_text與重建全文，避免把提供字元當成模型生成。
不會自動接訓練或下一個GPU工作。

## 一行純文字：先準備，未訓練

使用者建議已實作為`training/buyer_check_lines.py`的render／parser／逐欄score，僅CPU測試；本次GPU診斷不追加這個第三條件，避免擴大工作量與混淆原因。
示例，金額皆美分：

```text
CASH=1959 | ROW=offer:o1:P74 total=1960 held=0 val=5881 adj=5881 afford=0 eligible=0
CASH=1959 | ROW=board:seller_X:P74 total=1960 held=0 val=5881 adj=5881 afford=0 eligible=0
```

可以在同一行接多個ROW，或換行。board必須包含seller ID；offer由當前唯一offer ID辨識，不混淆不同賣家同商品。
parser容忍大小寫、額外空白，以及以逗號／分號／換行替代豎線。缺一欄或該欄數值損壞，只扣該欄，其餘可獨立得分。
重複衝突欄位只使該欄失效；錯來源／ID不能借用別列答案；缺候選仍計入分母。金額不自動換單位、答案不從observation回填。
只解析think草稿，不把工具內private_state冒充事前檢核；若缺think關閉也不從tool_call後偷抓答案。
這些都是量測慣例，不加引擎限制。單一格式錯誤不扣全部欄位，也不能因格式寬鬆就放過現金／持有的實質錯誤。
格式較簡短不代表模型看幾次必然學會；空入口結果出來後再決定是否改訓練目標，原JSON資料與實驗原樣保留。

CPU檢查：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/diagnose_self_check_entry.py --prepare-only
/home/hank/.cache/adl_final_venv/bin/python -m unittest training.test_self_check_entry -v
```

只做資料／tokenizer／parser與mock launcher，未啟動真實模型；CPU成功不是GPU結果。
