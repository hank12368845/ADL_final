# Seed2獨立核對與兩seed比較

結果：results/basic_checks/20261120-225505_buyer_pair_6xfah1ox。
兩場各450行動完整、非mock；來源／adapter／原persona／版本指紋與初始狀態一致，逐單位效用／支出／現金符合engine raw surplus。實際seed2，雙卡均有負載。無新訓練或部署。

| Persona | step84 surplus $ | step10 surplus $ | 差額 $ | 零效用支出84→10 $ | 錯掛牌84→10 |
|---|---:|---:|---:|---:|---:|
| p1 | 67.30 | 108.30 | +41.00 | 5.90→6.12 | 1→15 |
| p2 | 66.51 | 125.49 | +58.98 | 24.45→0.00 | 3→6 |
| p3 | 21.60 | 145.95 | +124.35 | 41.71→23.79 | 5→2 |

三persona帳目155.41→379.74（+224.33，144.35%），不是正式persona聚合。step10兩seed皆有較高帳目合計，p2與p3各自兩次均提升；p1 seed1 -3.01、seed2 +41.00。step10值得優先進官方Phase2 economy驗證，但還未完成正式本地規模選模。

## 操作退化的逐題證據

step10 p1在tick16看到P67實際有貨卻被同tick其他買家搶走，此次屬競爭。tick17正確換買P68成功；但tick18–30連續13回合，完全相同參數buy_listed(ta_seller_p3,P67,1)，即使當前board沒有該賣家／商品配對，仍每次收到no_stock。tick27有另一賣家ta_seller_p1掛P67，模型仍指向已無貨的ta_seller_p3，其餘這段回合沒有任何P67掛牌。
這13次是反覆選失效組合，不能算成同tick競爭輸掉。p1總共15次錯掛牌，p2另6次、p3另2次，合計9→23，尚未修好掛牌綁定與失敗後更新。

核對llm.messages最後一筆user：當前board與you逐欄等於事件observation，每回合prompt指紋各異，確有推論耗時及usage。不是報表重複列同一回合或應用端重用同一prompt。
模型tick18下單失敗後仍將P67加入自己的private_state持有清單，並把cash寫成308.16；tick19後真實you.cash仍339.11且未持有P67，記憶卻維持假成交、live=true、bad=[]，模型接著反覆下單。證據支持自寫記憶與實際結果不同步、未重新核對當前board；不把它宣稱唯一根因。

step84 p1截斷2次（tick3、27），連帶6次額度wait；step10本seed截斷0、額度wait0。但seed1 step10 p1曾整輪因loop耗盡額度，所以不能說已根治思考迴圈。

零邊際效用總支出72.06→29.91，step10仍有p1 tick9重買P22虧6.12、p3 tick18重買P58虧23.79；p2本場沒有重買成交。負收益單位總損失125.07→32.61；有利單位淨收益280.48→412.35。收益提升131.87加減少損失92.46，合計224.33；均為帳目拆分，不是隔離因果效果。
學生low grade單位8→1，品質交付與對手行為也隨市場分歧，不能只歸因於學會品質辨識。兩版本現金不足均0，低現金邊界仍未驗。

## 兩seed後的開發判斷

保留step84基線，step10為優先送官方Phase2 economy測試的候選；不自動修改部署。不要再無限加單場seed或操作小題，也不立即續訓。
下一階段應用run_economy.py --phase 2、隔離設定指定正確adapter、保留原persona與規則，評估3economies×5rotations的穩定收益；不得直接跑預設starter identity checkpoint冒充step10。正式GPU流程仍須準備進度／ETA與路由，交由使用者啟動，本次未啟動。
兩seed均為公共商品、toy對手、單rotation；候選兩次提升是有限正面證據，並不證明正式隱藏商品／對手高分或所有persona操作改善。
本次兩arm自此均為development已見；若未來回灌訓練，必須改標seen。截斷／失敗的機會成本不能直接換算成確定扣分，零效用支出已包含於surplus，不可重複扣除。
