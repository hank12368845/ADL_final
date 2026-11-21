# 明確指示仍未通過後：下一步建議

狀態：使用者已批准此下一步；診斷persona副本與36題命令已備妥，見BUYER_COMPACT_CHECK_V1.md。
現用persona、訓練資料與提交adapter沒有改動，GPU尚未由Codex啟動。
依據：results/basic_checks/20261120-144106_dollar_instruction_ux_83tdw/independent_instruction_review.md。

## 決策

暫停新CASH格式與loss微調，保留官方step84作固定權重；dollar6保留對照，不以12個seen的局部改善選為最強版本。
使用者要求的自主四要素檢核仍保留：cash、total、held、current next-unit value與是否合格。
但可核對的自然短句也能提供證據，不再以特定開頭／語法作改善本身。現有失敗門檻與紀錄不回改成通過。

## 下一個有上限的實驗：只改診斷persona，零訓練

固定step84，建立一套明標compact_check的三persona副本，目標約1000～2000字元／份。
現用三份6403／7600／7031字元原樣留存，不用短版冒稱原latest。
精簡重複規則、長篇示範與附加策略，讓核心順序保持：

1. 從當前board與offers找到真實候選，不先被初始估值／持有清單中的舊商品帶走。
2. 用短句核對當前cash、price×qty、是否持有、current utility；缺失／0 utility不拿初始價值補回。
3. 明確判cash>=total；正margin不能覆蓋現金不足或已持有。
4. board只用匹配seller/product與正庫存列；offer只用當前offer ID接受，不轉成虛構board。
5. 比較合格選項，沒有才wait，結束思考並輸出一個原生工具。

保留三persona的主要優先順序與品質風險處理；不額外訓練新議價策略、不提供正確數字或資格答案。
此為整體persona精簡對照，不會把長度／例子／規則位置等多種變化的效果硬歸為單一因果。
所有工具／觀測仍用正式原樣；private_state的事前事實／不可預先更新成交狀態要求保留。

本次只新增36次推論：原seen12作回歸＋未見來源development24，兩卡分流、同題／pair固定GPU。
父step84配原persona的36題已有原始cached結果，重用並明標，不再花GPU跑底座。
這24個validation已被前輪使用，不能稱封存test；能降低只在同12題上調整的風險，但仍需獨立市場驗證。

## 預先評估原則

- 最終動作：cash錯誤、持有交易、not_current_board、無工具、截斷都分開計數，追蹤同題修好／退步。
- 正例捕捉：相對固定step84，seen與validation各自保住，各persona另列；不接受全wait換安全。
- 自主檢核：逐題留cash／total／held／current value的原文證據，區分明確正確、明確錯誤、未呈現。
- 不從private_state回填成事前思考證據；不靠寬鬆解析消除實質數字／ID錯誤。
- 舊十欄private_state等自訂限制若保留，需與候選persona一致；任何改動先明列，不冒稱引擎要求或靜默放鬆舊門檻。

金融／操作改善與原有捕捉皆成立後，才進下一階段；不以格式出現率、單場合法率或scalar loss判最強。

## 下一階段：市場收益，而不是繼續追格式

條件改善才安排一場正式尺寸市場（5seller×10buyer、5round×6turn），用可執行的現用／候選對照確認surplus、成交數、浪費回合與品質損失。
先一場完整市場，不直接跑三economies／五rotations全部；GPU仍由使用者啟動，需提供場內與整體進度／ETA。
正式尺寸一場仍非最終成績或泛化證據；公開TA只是toy對手，最終還需要不同seed／商品／對手。
是否繼續SFT由操作與市場缺口決定，不再預設每次未過就換格式、加loss權重或續epoch。
