# Seller官方基線v1：完成後獨立審查

Run：`training/runs/20261121-135008_seller_official_9d95_ubx`。使用者完成訓練；Codex本次僅CPU核對來源、資料曝光、optimizer、權重與原始25×4份自由生成，沒有啟動GPU、重訓或部署。

結論：訓練有效改善了「輸出一個可用正毛利報價」與收束能力，step46保留實戰候選；尚未證明實際profit提高，庫存條件仍有缺口。下一步固定v2 persona與權重，比較starter／step46完整新市場，不先加epoch或改草稿格式。

| step | 官方未見NLL | 有貨正毛利報價／9 | 官方未見守則／6 | 全部25題截斷 | 全空庫存正確wait／3 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.8059 | 3 | 4 | 8 | 1 |
| 12 | 0.7380 | 7 | 6 | 1 | 1 |
| 23 | 0.7217 | 5 | 6 | 1 | 0 |
| 46 | 0.7099 | 9 | 6 | 0 | 0 |

Step46現行三persona分項均為3/3有貨報價，但各有1題全空庫存仍set_price。不是全wait塌縮；也不能因6/6官方未見、NLL下降或引擎接受就當高分。

## 確認沒有重踩的流程問題

- source/data/persona指紋、整市場split與固定25題完整輸入核對通過。當前v2三persona與p1v1控制確實使用，無訓練／測試persona偷換；封存test35未用於本次選模。
- 實際46/46更新，363個不同train樣本全部到位，尾batch補5筆共368曝光。逐rank trace與固定batch順序完全一致。
- warmup3步後正常cosine，全程不是只在warmup。三checkpoint optimizer中的所有參數步數分別12／23／46，權重全部finite、248個LoRA B tensors非零；step23與step46相較前存檔496個tensor均有更新。rank16／base／noDoRA／language-layer PEFT檢查通過。
- 最長反傳預檢已在GPU實際通過，32層checkpoint與32個分塊MLP有效，實際選到FLA／causal。預檢peak11.18GiB，訓練metrics最高約11.71GiB；沒有無加速對照，不能宣稱快幾倍或直接調低未驗證顯存門檻。
- 全流程約44分59秒，含底座與三存檔共100次自主推論及validation，不全是optimizer計算。`training_seconds`含存檔／中途及最終評估，不能當純訓練吞吐。
- 實際GPU0單卡，run記錄selection='0'；當時GPU1 free18.32GiB，扣2GiB後也未達本輪18GiB訓練預算。globalbatch仍8，沒有少看一半資料。

## 需要修正的報表解讀

原`current-rule valid`是開發守則，不能直接當引擎合法率。依官方`Market.apply`用相同控制盤面重新執行：step0／12／23／46為4／11／10／12個行動接受（各12題）。原報表保留不改，獨立審查另存。

- `set_price`即使商品目前0庫存，引擎仍接受；沒有商品可賣時，這是無法促成成交的行動，不是`no_stock`工具拒絕。`make_offer`才會因不足庫存被拒絕。
- step46 p1一題`private_state`1138chars。正式`LLMPolicy`會截為1000chars，並不因此拒絕整筆工具；仍有輸出token及記憶截斷風險。
- 全空庫存題是人工round1／無history的極端盤面，不代表自然市場25%失敗。整個庫存都空後本來就沒有剩餘商品利潤，不能編造確定profit損失。更需在市場檢查「混合庫存時選錯已售罄商品」是否拖累其他有貨商品。

獨立重核程式`training/audit_seller_baseline.py`只讀原始產物與在CPU執行引擎，不改原訓練程式／dataset／checkpoint，故保留resume與重現來源指紋。所有檢查通過，證據`/tmp/adl_seller_baseline_independent_audit.log`。

## 尚缺的能力與證據

1. 資料沒有全空庫存樣本，363 train只有4個wait，是已確認覆蓋缺口。模型在草稿或筆記寫出「all stock0」卻仍掛牌，不能假設只要寫對文字就會選對工具；也不能未做對照就說覆蓋是唯一成因。
2. Step46 p1 stocked題掛P03=2.00，而此控制世界的買家高品質效用=2.00，買方surplus為0；正毛利報價9/9不等於能成交9/9。私有買家數字只供CPU診斷，未寫入模型輸入／persona，不過擬合公開ID。
3. 其餘兩persona仍有按商品順序先上低價品的傾向。是否能提早曝光高毛利商品、避免無效重掛、控制品質聲譽與清倉，要由動態市場驗證。
4. HF小題是greedy、單題1024上限；正式市場是vLLM、temperature0.7／top_p0.95、每round8192，不能把底座小題截斷8次說成原starter市場一定有8次。正式6history穩定度尚待驗。
5. 官方未見只有6題／3個市場group，train24商品僅1筆，未證明private商品或正式TA對手泛化。

目前決策：保留step46及step12／23，starter基線不變，無自動升級。直接準備同persona、同初始條件的新seed完整5seller×10buyer、30turn成對市場，對比各persona實際profit、售出／未售、報價與實現毛利、無貨重掛、截斷與低grade評論代價。這份結果不足以合理要求另一輪格式小修或直接續訓。

先讀run內`independent_seller_training_review.md`，完整核對含原始草稿、逐題engine結果、資料／權重證據在同目錄`independent_seller_training_audit.json`。
