# 待驗證的美元檢核格式指示

step6未過門檻後的下一步建議，僅作可審查草稿；沒有改agents中的現用persona或任何提交權重，沒有啟動GPU。
應先以非提交副本測「原persona＋此段」的同題純推論，確認格式遵從與實質判斷後再決定是否整合與訓練。
此條件與原persona不同，必須標記為format_instruction診斷，不能冒稱原始latest的自由生成改善。

以下指示329字元，沒有填入任何題目金額、實際商品／offer ID或資格答案：

```text
Start thinking with this dollar check:
CASH=<cash> | ROW=<board:seller:product OR offer:id:product> total=<price*qty> held=<0/1> val=<current next utility> adj=<expected value> afford=<0/1> eligible=<0/1>
afford=1 iff cash>=total. eligible=1 only if unheld, live, affordable and adj>total. Then finish thinking and call one tool.
```

以兩個換行接在各原persona後，總長分別為6734／7931／7362，均低於8000字元。
這個長度檢查不等於模型遵從證據。若真正整合，要同步訓練與評估使用同一版本，保留原文及SHA。
先驗同12題的三個差一分負例、三個持有負例、六個正例，逐欄與工具／state分開評分。
格式通過但afford錯誤，不算完成；門檻不因新格式或新的persona條件而靜默放寬。
