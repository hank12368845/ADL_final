# 原persona的裸底座vsstep84結果（2026-11-20）

結果：results/basic_checks/20261120-153410_base_step84__fme0dw_。
先讀該目錄independent_base_step84_review.md／json；原報表base_step84_review.md／json已標記語意審查完成。
原自動JSON保存在base_step84_review_before_semantic.json，分數未改。semantic_review.json原pending證據仍保留，
semantic_review_completed.json為36新份完整閱讀＋36cached精確比對後沿用前輪標註；非獨立人工gold。

兩卡各18，360.05秒，exit0，零訓練。來源／版本／prompt／cached權重／路由／EOS與重算分數皆吻合。
底座→step84：通過16→19/36、捕捉14→13/18、負例安全2→6/18、both_correct1→5/18、
cash5→7、owned5→9、非當前掛牌2→0、無工具7→1、截斷8→1。
底座有1題雖解析到正確工具仍截斷，所以無工具與截斷數不同。

LoRA有操作收束與部分安全改善，非全面勝出；尚無收益優勢證據。
底座相對step84少2個cash錯誤，包含3筆step84的cash錯誤改成底座截斷、另一筆owned改cash，不能只看此數字說底座更會管錢。
底座真正修好3題、退步6題；兩筆p3有利offer底座接受，step84卻重買P72／P60，為實際弱項。
重買在兩者都有，不斷言全由SFT造成。p1／p2 step84較穩、p3仍弱，不硬將小題結果當市場成績。

裸底座cash明確正確36/36、四要素完整正確7/36，仍可能轉追catalog並截斷；草稿與最終工具必須分開驗收。
題庫為現金／持有介入快照，部分原budget／可見歷史與覆寫cash不一致；當前cash仍具權威，但不能把這批題當自然市場分布。

工作決策：保留原persona＋官方step84作下一場5seller×10buyer、5round×6turn市場診斷基線，
裸底座對照保留。停止格式／persona／小輪SFT，先量實際surplus與每persona失誤成本，再決定訓練方向。
這不是最終選模或最高分保證；本次未部署、未執行市場、未啟動其他GPU工作。
