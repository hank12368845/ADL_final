# Buyer Operations v2：先修復買／等決策

## 為什麼改

v1 step8 在 vLLM 及 HF／PEFT 的正例均全wait。HF對照run為
`results/basic_checks/20261118-213117_buyer_hf_p4c2qdho`：8/8 wait、0次正收益捕捉，
兩卡各4次、無解析失敗／截斷。正確工具名稱NLL由底座0.0123升至0.9953，
wait名稱NLL由7.5586降至0.1150；交易參數NLL雖低，卻是給定正確工具前綴後的條件預測。
足以將開發重心移回資料／訓練目標，尚非所有backend數值完全一致或某單一SFT因素的因果證明。

本版同時改資料、sampler、loss及短試驗LR，用於驗證修復可行性；
成功也不能宣稱某一項改動單獨有效，後續需要時再做消融。
沒有修改正式engine、目前agents中的persona或既有adapter。

## 目前直接執行的命令

資料已由CPU產生，使用者在專案根目錄啟動雙卡訓練：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_operations_v2.yaml --gpus 0,1
```

從原始底座建立新LoRA，不載入v1 step4／step8，不自動部署或接著跑市場。
啟動前依各卡剩餘顯存扣2GiB設定預算，訓練預算不足18GiB則跳過；
顯式要求兩卡而不足會停止。獨立模型副本採DDP同步梯度，各卡microbatch1、累積4次，
global batch8。不得把兩張卡的零碎顯存當成同一張卡，也不終止他人的工作。

終端顯示模型載入、最長序列前後向preflight、validation與訓練step／micro進度及ETA。
載入期間ETA估算中；不保證雙卡恰好兩倍速度。HF缺少加速kernel時可能較慢。
`--cpu-check`不查GPU／不啟動訓練；`--plan-only`會查GPU但不訓練。

## 資料設計

CPU入口 `training/buyer_operations_v2_data.py`，資料版本與v1分開保存。
新seed為20261119，與v1的市場seed不重疊。40個市場先切32／4／4組，再展開三persona及正反例。

| Split | 市場組數 | 筆數 | buy / accept / wait |
|---|---:|---:|---|
| Train | 32 | 1,536 | 384 / 384 / 768 |
| Validation | 4 | 192 | 48 / 48 / 96 |
| Test | 4 | 192 | 48 / 48 / 96 |

每個市場／persona含8對決策：board及offer的現金、估值、持有量，
另有board無庫存及虧損多件offer。每對共用system prompt、歷史與備忘錄，
只介入當前決策條件及其必要的衍生觀測（如offer文字同步價格／數量）。
不是額外的contrastive loss；是成對標註的SFT資料。

正例為單件、固定品質、可負擔且正surplus，目標buy_listed或accept_offer；
負例的同一交易會被引擎拒絕或產生負surplus，目標wait。
現金／價格負例同時涵蓋明顯差距與一分錢邊界。每筆答案由原Market.apply驗證，
價格高於效用的負例維持可負擔、合法成交而虧損，避免與現金不足混淆。
額外嘗試負例交易以確認拒收或實際虧損。答案／label／驗證結果不進模型prompt。
零現金屬合法狀態，v2教師會直接等待，不套用v1的「正價格」轉換器。

這些是有原引擎backing state的受控反事實資料，包含現金／持有狀態介入，
不聲稱每一筆都是完整可達的15-agent市場軌跡。
目前是單一候選、固定品質的操作課程；尚未涵蓋多掛牌排序、品質風險、談判及長局資金配置。

約40%的pair移除歷史，正反例一起移除；其餘保留由原引擎實際執行的
buy_listed／accept_offer／wait各一個。三動作順序按市場／persona輪替，
最後一個動作也會變，避免只平衡總量但最近動作仍總為wait。
fresh仍保留當前結果與最小備忘錄，不假裝是從未交易的市場。

## 每個global batch的組成

`training/buyer_operations_sampling.py`固定每步：

| 目標 | 數量 |
|---|---:|
| buy_listed | 4 |
| accept_offer | 2 |
| wait | 2 |

每步至少包含一對board及一對offer完整正反例，負例family輪流出現。
另補足正例後打亂順序，同一batch不重複樣本；不同step可重複取樣。
因此原始資料wait占50%，實際訓練sampler占25%，兩者不可混淆。
16步共128次樣本曝露＝64 buy、32 accept、32 wait，並非完整epoch。
三persona每批以3／3／2輪替分配，16步共43／43／42次，避免短試驗過度偏向其中一組。
固定seed可重建全域順序，兩卡各取每批交錯的4筆，梯度平均後等價於global batch8。

`batch_plan.json`保存啟動計畫；實際worker寫`samples_rank0.jsonl`／`samples_rank1.jsonl`。
每步合併兩卡實際action counts再次檢查4／2／2，不符合則停止；
`finished_rank*.json`保存各卡完成步數、樣本數與顯存峰值，供完成後核對真實分工。
CPU重建兩rank順序通過不代表GPU已實際執行。

## Loss與訓練設定

```text
L_completion = mean(最後assistant答案每個token的CE)
L_tool_name  = mean(該答案function名稱token的CE)
L_total      = L_completion + 1.0 × L_tool_name
```

名稱位置依官方Qwen XML模板及token offset定位，索引相對於最後completion。
prompt、工具結果、歷史assistant動作全mask；private_state不被誤算為工具名稱。
兩個loss各自取平均，工具分支不再只占長JSON的少數token權重。
不額外提高參數loss；目前證據優先支持修復工具分支。

v2配置：rank16、alpha32、dropout0、AdamW、LR1e-5、weight decay0.01、warmup2、cosine decay、
16 optimizer steps、global batch8、bf16、SDPA、gradient checkpointing、thinking OFF。
底座及tokenizer pinned revision沿用v1，僅訓練語言層LoRA，PEFT safetensors。
最長12288 tokens，超長直接失敗，不截斷事實，也不只丟掉正／反其中一筆。

每4步保存並驗證，產生step_0004／0008／0012／0016及final_adapter。
內嵌validation使用16筆、8個完整正反例pair、涵蓋八個family。
分別記錄completion、tool_name、objective及各動作loss。
v2訓練log的loss是加權總loss，不能與v1的純completion loss直接比較；
資料也已改版，loss比較需固定題目與定義。
**這是teacher-forced loss驗證，不是生成行為選模，也不自動宣稱最佳checkpoint。**
metrics的`lr`是本步實際使用值，`next_lr`是scheduler更新後值；
舊v1歷史log的lr是更新後值，不回寫舊結果。

## 訓練後的行為檢查

先由使用者針對checkpoint執行小型配對驗證，例如：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_buyer_operations_v2.py \
  training/runs/<本次run> --checkpoint step_0004 --gpus 0,1
```

也可用step_0008、step_0012、step_0016或final。每次預設192calls：
16題×三persona×兩seeds×base／candidate。兩模型在相同輸入、persona、seed及GPU配對。
probe來自兩個validation市場的board／offer現金及估值正反例；完整validation_cases.json
另有四個市場、全部八family，共64題，待小試驗改善後再擴大。
test市場不進開發probe。兩個模型直接比較新題，不能拿與v1不同題型的分數當作提升證明。

先讀operation_gate.md，分開看各persona的正例捕捉、負例正確等待、格式／參數錯誤及記憶錯誤；
operation_gate.json另外記錄交易precision，零交易為N/A，不是100%。
全wait、全買及少數格式進步均不足以證明學會決策。先確認買／等有區分，
再比較是否優於相同輸入的底座，通過後才擴大訓練與正式市場surplus驗證。
短probe不能保證三個persona的正式grade或多對手泛化。

## 從零重現與檔案

```bash
PYTHON_BIN=/home/hank/.cache/adl_final_venv/bin/python bash training/download.sh
/home/hank/.cache/adl_final_venv/bin/python -u training/buyer_operations_v2_data.py
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_operations_v2.yaml --gpus 0,1
```

資料目錄已存在時生成器拒絕覆蓋；目前已備妥，不需使用者重建。
若要重建新版本，指定新的--output-dir並調整新config的data_dir，保留既有證據。
manifest凍結資料／生成器／supervision依賴hash；run記錄training_sources、資料manifest hash、
套件版本與GPU選卡snapshot。raw/tokens、CPU review與雙卡預覽在training/data/buyer_operations_v2/。
先前CPU生成草稿另存，未供本版配置使用。

新run在training/runs/<time>_buyer_operations_*/，保存runner.log、train_config.json、
batch_plan.json、metrics.jsonl、硬體／rank sample trace、checkpoints與final_adapter。
尚無v2 GPU訓練時間或收益；使用者執行後依實測補上。
尚未指定最終seller／buyer提交checkpoint，candidate不自動安裝至agents。
不支援完整optimizer resume；中斷保留已完成checkpoint，重開預設仍從底座開始。
