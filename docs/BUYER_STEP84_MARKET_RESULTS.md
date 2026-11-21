# Step84完整市場結果與下一步優先序（2026-11-20）

結果目錄：results/basic_checks/20261120-160558_step84_market_st8d6o1p。
先讀run內independent_market_review.md／json，原自動帳單market_loss_review.md／json及engine scores保持不變。

完整450行動、每persona30，耗時20分35秒，真模型無infra／policy repair問題。CPU重算全部逐單位效用與scores一致。
p1 surplus192.34（第2/10，21筆／20種），p2 141.65（第6，11筆），p3 143.29（第5，12筆）。
三人合計477.28、pooled mean159.09，只是本場帳目，不是正式persona聚合或grade。

原先疑慮的實戰帳：唯一零效用成交為p1 tick30重買P19、直接損失5.40；p3重買成交0但仍有1次已持有P27的失敗嘗試。
現金不足0，期末現金仍199.15／279.46／305.10，未壓力測到一分錢邊界。全成交high grade，品質避險亦未驗證。
p2 tick3截斷7526tokens後被迫wait3次；p3 tick23截斷7155後wait1次，共6turn。
兩個loop占學生生成tokens38.5%；p2只有2history／10690input就loop，不能歸因為長上下文上限不足。

更大的缺口是11次當前掛牌不存在（p2 7、p3 4）：6次商品存在但seller配錯，5次商品整個board不存在。
p2 tick20～23四連錯把ta_seller_p2的P60讀成ta_seller_extra_2。p2／p3嘗試購買25／24次，非單純不敢買，只有11／12次成交。
另23次是同tick真實搶光；13次發生於stock2或3，不能只用stock>=2避碰撞；其中2次原本就重買，不能當全部都是正收益流失。
雙卡實際233／204次請求、90275／121195生成tokens，確實分流。

接下來優先當前來源seller/product綁定＋失敗後換候選，以及重複生成的工具/EOS收束；保住正收益捕捉，再處理競爭策略。
cash／held維持回歸，不再為固定格式／一分錢題反覆短SFT。本場保留baseline，不直接回灌再當未見驗證；後續資料按獨立來源／整市場切分。
本次只完成分析，未開新GPU／訓練／改persona／部署。單一public市場和toy TA不構成高分保證。
