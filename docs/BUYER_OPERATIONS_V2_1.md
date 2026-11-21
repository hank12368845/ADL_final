# Buyer Operations v2.1：現金邊界與 offer 召回

本版保留v2 step8為高召回對照：同題正例47/48、offer23/24，step16則降至32/48、offer9/24，
同時負例正確等待17/48→34/48。目標是改善現金條件判斷並保住有利offer，
不把這些操作診斷數字當正式市場surplus。

## 現在執行

資料已由CPU產生，使用者只需啟動雙卡訓練：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_operations_v2_1.yaml --gpus 0,1
```

從原始底座建立新LoRA。v2 step8僅作評估對照，不是訓練初始化／optimizer resume來源。
維持v2的rank16、alpha32、LR1e-5、warmup2、global batch8、16 steps、thinking OFF，
以及原loss：completion CE + 1.0×獨立平均的tool-name CE。未新增記憶loss或修改persona。
每4步保存，產生step_0004／0008／0012／0016及final_adapter。

兩卡DDP各microbatch1、累積4次，規劃各64筆；啟動前依剩餘顯存扣餘裕，明確要求兩卡不足則停止。
終端顯示載入、preflight、validation、step／micro進度與ETA；尚無速度時顯示估算中。
輸出新training/runs/<time>_buyer_operations_*/，保存log、分項loss、batch_plan、各rank實際sample trace、
GPU snapshot、source hash及完成紀錄。既有checkpoint保留，不自動部署或接著評估。
尚無v2.1的GPU時間或收益，不能把CPU通過當成策略改善。

## 資料改變

沿用v2的32／4／4市場split，維持held-out邊界。每個pair完整保留fresh與recorded兩版本，
只移除歷史，保留相同的當前觀測、備忘錄與上次結果。raw筆數為3072／384／384。
不是把曾看過的validation／test失敗題塞進訓練。

現金pair以固定品質商品、同一P／V建立掛牌及binding offer兩種表示：

| Variant | 正例現金 | 負例現金 |
|---|---:|---:|
| equal | P | P−0.01 |
| above_one | P＋0.01 | P−0.01 |
| wide | 原本充足現金 | 約P的一半 |
| zero | 原本充足現金 | 0 |

每個市場的variant跨persona／channel共享；每個split都涵蓋四種variant。
所有正例原引擎合法成交、surplus>0；現金負例的交易嘗試必須得到insufficient_cash。
其他估值／已持有／缺貨／虧損多件offer正反例沿用凍結v2生成器，避免只教現金而忘記其他限制。
答案、label、variant及economic_id只在資料metadata，不進prompt。

這是有原引擎backing state的受控狀態課程，不是完整15-agent市場軌跡，
目前仍為單候選、固定品質教具；不宣稱已學會多掛牌排序、議價或品質風險策略。

## 取樣改變

每個global batch維持4 buy_listed／2 accept_offer／2 wait，三persona按3／3／2輪替。
進一步固定每個動作在兩種context各占一半：

| Target | fresh | recorded |
|---|---:|---:|
| buy_listed | 2 | 2 |
| accept_offer | 1 | 1 |
| wait | 1 | 1 |

16步共64 buy、32 accept、32 wait，persona43／43／42；fresh正例offer16筆，
v2先前實際只有8筆。每步保留board及offer各一個完整正反例pair；
現金pair在兩channel選相同市場與variant，使金融條件一致。
32筆wait中現金不足16、估值8、持有量4、庫存2、多件offer2。
四種現金variant均在fresh／recorded出現，不把context當成買／等提示。
128次曝露並非完整epoch，跨step可重複取樣。

內嵌16筆validation使用完整pair、八種family與平衡context，仍是teacher-forced loss檢查。
它不取代生成評估。原始v2資料／sampler保留，v2.1使用獨立生成器與sampler。

## 訓練後比較

先用完全不變的舊16題，直接比較v2 step8與新candidate，預設192calls：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_buyer_operations_v2_1.py \
  training/runs/<本次run> --gpus 0,1
```

可加--checkpoint step_0008等挑已存checkpoint。兩者相同persona、題目、seed、GPU配對，
本版不重跑plain base。operation_gate.md／json會明寫reference_model=step8，
不把該LoRA叫作原始底座，並單列正例offer捕捉。
若對照checkpoint在訓練準備後被改動，fingerprint檢查會拒絕評估。

再用新現金控制測試，一個held-out validation市場×兩channel×四variant×兩context×正反例，
共32題、384calls：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_buyer_operations_v2_1.py \
  training/runs/<本次run> --suite cash --gpus 0,1
```

先檢查三persona的正例／offer召回有無保留，再看unaffordable及負例等待是否改善；
記憶格式作次要診斷，不以該分數直接當引擎扣分。這些是開發validation，test市場仍保留。
相同seed也可能有生成波動，不由少量翻轉宣稱正式grade或單一根因。

## 重現

底座、套件與下載流程沿用training/README.md，不需新增套件或私有憑證。
從零準備v2.1 LoRA只需download.sh、buyer_operations_v2_1_data.py，再執行本版config。
生成器拒絕覆蓋既有目錄，目前本機已產生，不需重做。
重現LoRA本身不依賴v2 step8；若要重現對照，先以v2流程產生自己的step8，
再把comparison_adapter指向該checkpoint。評估需要對照存在，訓練不使用其權重。

CPU檢查包含：原引擎現金等價／一分錢／零現金、board／offer金融條件一致、
動作×context與persona配額、真實DataLoader雙rank重建、官方token mask、
舊probe完全一致、假worker完整step8/candidate路由報表、reference變更拒收。
假worker或tiny CPU model不代表真模型效果。實際GPU訓練／評估均由使用者啟動。

## 已完成 legacy 後的三階段對照

本輪v2.1 final比v2 final正例32→36：掛牌淨增1、offer淨增3；
但負例等待34→29、非正收益成交3→8、cash錯誤仍11，屬局部改善。
11次cash失敗為6次同一一分錢場景、3次明顯不足、2次零現金。
下一步用同一命令依序跑final cash及v2.1兩個中間checkpoint：

```bash
bash training/evaluate_buyer_v21_followup.sh \
  training/runs/20261118-233240_buyer_operations_pa514zcv --gpus 0,1
```

共768calls（384＋192＋192），各階段都有進度／ETA及獨立結果資料夾，失敗即停止。
中間step_0008／0012是v2.1 candidate，對照仍是v2的高召回step8。
可加--port 8100；此wrapper只接受GPU／port選項，避免誤把prepare-only當完成推論。
先看cash=P及P＋一分的正例是否被錯拒，再看不足一分／明顯不足／零現金的負例，
並分開board／offer及fresh／recorded。選模後再用少量完整市場比較surplus，尚不進行新一輪長訓練。
