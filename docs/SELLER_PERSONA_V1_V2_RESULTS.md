# Seller persona v1/v2 真實市場結果

2026-11-21：`results/basic_checks/20261121-005859_seller_personas_1f4wxek7` 已完成兩場。CPU 重新執行 `compare_seller_personas.py --report`，來源指紋、相同初始市場驗證、每場 450 行動及成交價減實際 grade 成本的逐筆帳目均通過。沒有訓練、GPU 啟動或部署變更。

| persona | v1 profit | v2 profit | 差額 | 售出 v1→v2 | 未售 v1→v2 |
|---|---:|---:|---:|---|---|
| p1 | 175.27 | 150.51 | -24.76（-14.1%） | 23→16 | 24→31 |
| p2 | 238.21 | 234.64 | -3.57（-1.5%） | 33→34 | 17→16 |
| p3 | 171.00 | 279.05 | +108.05（+63.2%） | 26→43 | 23→6 |

三 persona 帳目合計 584.48→664.20（+79.72／13.64%），不是正式 persona 聚合分數。三份 persona 同時改變、市場互相影響；單一 seed、單一 stock rotation、公開商品及 toy TA 買家，不將差額視為各 persona 的獨立因果效果。兩版都使用未訓練 starter identity adapter、Thinking ON；尚無 Seller 專項 LoRA 成果。

## 實際行為與失分

- p3 明顯改善在成交量，而非已證實賣得更貴：每件平均實現毛利約 6.58→6.49。v1 全場 30 次 make_offer，v2 則 23 次 set_price／7 次 wait，v2 沒有 make_offer／send_message／房間發言。不能因 persona 寫了私人說服與 bluff，就宣稱它真的使用這些技巧並因此賺更多。
- p3 新版兩件 P22 是 low grade，每件 5.50、成本 2.00，總 profit 7.00；相同成交價下相較 high grade 共省成本 4.00。兩買家分別給 2 星與 1 星。不是沒有聲譽代價，也不能用 4.00 的成本差解釋 108.05 的總提升。
- p1 v2 首先按商品表順序上架低價產品；前兩 round profit 只有 3.98。全場 28 次 set_price，9 次重複相同商品／價格／grade 的既有報價（不是每次都已證明完全無作用），最終 31 件未售。沒有在本場輸出驗證其高錨定／稀缺高價策略；成交商品收入恰為相同組合的 list price 總額。
- p2 v2 全 30 回合 set_price，沒有實際使用房間協調或私人折扣，與 v1 收益接近；不能將 persona 文字裡的價格領導視為已落實。
- 所有學生 seller 兩版均無工具拒絕、成本以下或等於成本報價、截斷、token 額度等待；所有 agent 服務失敗為 0。操作穩定不等於策略已充分實現。

v1 耗時約 20 分 57 秒，兩端點實際有負載；v2 耗時約 34 分 29 秒，資源規劃與實際負載為 GPU0 單副本。先前 v2 啟動失敗的 run 已歸檔，本次完整 run 為 `v2/run_market_20261121-125021`，未混入舊失敗成績。

## 後續判斷

保留兩版，p3 v2 是收益候選；p1 v1 暫有較佳本場成績，p2 兩版接近。不能把各版最高值相加當成已測混搭成績，也不據此自動換部署。

Checkpoint1 迫近，優先把官方 Seller 資料審查與第一個可重現 LoRA 基線補齊。資料需保留現用 persona、工具及真實成本／庫存語義；驗收同時看 profit、售出／未售、單件毛利與策略實際行為。p1 的商品優先序、重複掛牌及收尾成交能力值得補強，但不以單場結果就大幅改長 prompt 或宣稱必然改善。

獨立分 round profit／重複相同報價計數另存 `docs/SELLER_PERSONA_V1_V2_RESULT_AUDIT.json`；完整比較見原資料夾的 `persona_comparison.md/json` 與各版 `seller_review.json`。
