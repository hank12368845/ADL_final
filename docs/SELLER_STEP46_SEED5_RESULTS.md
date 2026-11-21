# Seller step46完整市場：seed5結果與下一步

來源：`results/basic_checks/20261121-151510_seller_models_87fz7bkq`。兩場真模型完整450行動，來源／初始世界／v2 persona／對手／正確adapter核對通過，服務失敗0；CPU重新逐筆計算成交價減實際grade成本，與官方帳目一致。

| persona | starter profit | step46 profit | 差額 | 售出starter→46 | 單件毛利starter→46 |
|---|---:|---:|---:|---|---|
| p1 | 200.43 | 236.81 | +36.38（+18.2%） | 24→25 | 8.35→9.47 |
| p2 | 219.90 | 215.80 | -4.10（-1.9%） | 28→25 | 7.85→8.63 |
| p3 | 274.31 | 225.59 | -48.72（-17.8%） | 35→35 | 7.84→6.45 |

本場沒有全面勝出，不能直接升級step46。三persona帳目合計694.64→678.20，差-16.44，非課程persona聚合；只一seed／rotation、public商品與toy對手，不能推斷整體顯著退步或正式分數。

## 哪些改善有落實

p1確實有高價成交：P55同售3件，單價22.50→26.00，profit增加10.50；其他P40／P67／P30／P60也有同量較高成交價。不是只有喊價變高。

同價且同grade重掛計數由p1／p2／p3的12／9／7減至0／3／0。這是實際行為改善，但沒有證明每個省下的回合都變成利潤，與無貨動作也可能重疊，不重複換算損失。

## 為什麼p3退步

兩版同售35件，主要可見差異是部分商品成交價較低。P48兩版同售10件，16.95→15.00，帳面profit少19.50；P62同售3件，26.97→22.00，少14.91，兩者合計34.41。P63同售1件，27→22，再少5.00。

step46第一個turn就把P48掛15，第三turn把P62掛22；不是先高價遭拒後才清倉。這些是實現價量的會計對照，不保證在step46其餘對局行為下維持starter價格仍能賣出相同量。

官方SFT這輪監督的是「成功執行、正毛利的工具答案」，沒有依實際成交profit排序或加權。成功set_price／make_offer只代表建立掛牌／offer，不代表賣出。319個教師報價中151低於list、105等於list、63高於list；低於list本身不一定錯，也不能憑這個比例斷言是唯一因果，但不能將操作SFT當成profit最佳化已完成。

## 小題0截斷沒有泛化到長歷史

step46 p2 tick10（round2）輸入17279tokens／6history，反覆寫同一組已掛牌商品，消耗當輪剩餘5707outputtokens，無工具；tick11／12自動budget wait。

step46 p3 tick16（round3）輸入23219tokens／6history，消耗剩餘4608tokens，`truncated_call=set_price`，工具未完成；tick17／18自動budget wait。該記錄只留548chars reasoning，沒有完整半截arguments，不能斷言一定是private_state重複。

兩個截斷共連帶4個budget wait；starter本場沒有截斷或budget wait。正式市場的input長度／history／採樣均與訓練最多8192tokens／2history及HF greedy1024小題不同；這是確認到的覆蓋差異，不是已證唯一病因。Seller的既有board在等待期間仍可被買入，所以不能把這6個無新動作turn叫作6次必然零銷售，或直接編造成確定profit損失。

兩場皆單端點GPU0，starter約37分02秒、step46約30分33秒，總市場執行約67分35秒，不含排隊。學生output總量53247→65992，不能用整場wall time較短就說step46生成更省tokens或加速多少。

## 修正庫存診斷：決策時快照與同tick成交

原報表將event.observation.you.stock直接當決策快照，有誤。官方`Market.observe`對seller回傳stock字典參照，稍後隨機執行同tick交易後，event寫出時可能已減少；board和已序列化給模型的user消息仍是決策時內容。

新`training/seller_market_observations.py`讀取實際送給模型的user JSON，並以initial_stock減去較早tick交易逐欄交叉核對。沒有改引擎、分數、權重或輸入；沒有model消息的mock／budget wait則用同一交易帳重建。

校正後「決策時已知庫存不足仍報價」：starter三者都是0；step46 p1=1（tick9/P72）、p2=0、p3=1（tick13/P60）。starter p3 tick15/P63是實際輸入有1件、執行時被同tick買走而no_stock，不能算模型漏看庫存。step46 p2兩筆原標記亦是決策時有1件。

原生成報表及凍結來源保留，正確逐題證據另在原資料夾`independent_market_audit.json`。下一輪報表直接使用校正方法，不再誤把競爭或同tick庫存變動當作已知無貨。

## 已批准的下一步：只補既有step12一場

保留starter收益基線與step46候選，不續訓。以同seed5、同v2、同對手，只新增step12一場450行動；重用已完成900行動並逐檔核對快取。理由是step12已有操作改善、尚未完整模仿到step46，檢查是否保住p3毛利；較早不保證更好。

這是既有checkpoint選擇，seed5已成為development依據，不再稱未見泛化。每角色只能交一個LoRA，不能把p1的step46與p3的starter拼成一個已驗證可提交模型。

具體指令與CPU驗證見[SELLER_STEP12_MARKET.md](SELLER_STEP12_MARKET.md)。本次Codex只審查／準備，GPU仍由使用者啟動，不自動部署或測更多checkpoint。
