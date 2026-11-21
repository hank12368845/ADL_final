# 官方 Buyer 資料基線：先固定實驗，再依證據調整

2026-11-19。依使用者要求，完整市場長測先暫緩，優先建立官方資料基線。
2026-11-20最新狀態：本輪112步訓練已完成，先做修正EOS的既有權重重評；不要直接重跑訓練命令。
命令與結果見training/README.md頂部、Guideline第36.6節。
先前合成資料修補、native sanity與市場wrapper保留為歷史對照，不再當作主要訓練來源。
本流程已完成資料下載、審查、CPU處理與測試；尚未啟動GPU，沒有預先宣稱能提高成績。
上一句是準備時的歷史狀態；實際訓練結果與評估問題以上方最新狀態為準。

2026-11-19啟動修正：使用者首跑 `20261119-212633_buyer_official_su7va01r` 在最長樣本預檢雙卡OOM，
尚無optimizer更新或可續跑checkpoint。HF載入後decoder維持eval，當時預檢漏了 `model.train()`，
因此設定好的gradient checkpointing未實際執行。現已在預檢前明確切換train並檢查每層checkpoint設定；
新增CPU混合attention小模型測試，驗證實際checkpoint函式被呼叫、loss一致且LoRA有梯度。
修正只影響執行模式，資料、LR、batch、長度與112步配方不變；由使用者重新執行原命令建立新run。
實際24GiB卡的修正後峰值顯存仍須重跑預檢確認，不把CPU通過稱作GPU OOM已實測消失。

## 先從過去的失敗固定方法

| 前幾輪問題 | 這次安排 |
|---|---|
| 未先使用官方訓練資料 | 官方資料為唯一訓練來源，固定commit與檔案SHA |
| 少量樣本、8／32步、warmup幾乎占滿 | 全部890筆curated訓練資料跑完整1 epoch，112steps，warmup6steps |
| 一輪換模板、一輪改配比，很難判斷原因 | 固定資料、切分、persona、loss與配方；第一輪結束前不做sweep |
| tool-name loss在gold答案後已接近零 | 使用正常assistant completion CE，沒有額外tool-name loss |
| 只看到loss下降，卻不會自由決策 | step0與每個checkpoint都做HF原生自由生成，分訓練題／官方未見題／當前規則防呆 |
| 補一邊、破壞另一邊 | 固定正例／現金防呆比較；大幅退化自動停、保留中間權重，不硬跑完 |
| 不同persona／市場混在一起說成功 | 原始輸出與各題錯誤保留；模型真正強弱仍需之後同條件市場surplus |

## 官方資料已實際下載並審查

來源：[miulab/ADL26W-final-train](https://huggingface.co/datasets/miulab/ADL26W-final-train)，
固定revision `fa0f94cff3eeed5bc54da597ad5c2a45ef491e45`。
Parquet檔87,733,052bytes，SHA256 `85e4824b750cb42d77513c16759b0860ee90b1d0e55b1c1cb87e184a29dfa815`。
公開下載不依賴私人token；原始檔保存於training/data/official_raw，包含seller資料供後續使用。

實際數量：1,091條buyer與1,208條seller軌跡。Buyer共有34,989次assistant決策，
其中7,466次有reasoning_content，包含真實多候選board、私人offer、議價、評論與房間協調。
官方資料有失敗動作；不是每一步都適合當正確SFT答案。

版本差異也已確認：所有buyer原始效用表都是舊版多件邊際效用、可延長的horizon；
包含6／12／24商品，以及私人／共享估值市場。有些舊persona寫沒有public board。
不能直接把整包資料宣稱為現行5round、單件效用的完整對局，也不能悄悄改觀測數字假裝一致。
這次保留原始system／persona／世界事實，只篩可轉移的動作目標；這項domain shift會列入基線限制。
其中128筆訓練樣本的原始persona仍含no-public-board字句，不在此版混入自行改寫persona的效果。
正式推論使用當前TA規則與三份乾淨短persona，不能沿用舊system當正式規則。

## ON基線資料怎麼選

第一版配合目前thinking ON，只取官方真的有短思考的成功工具目標，沒有替OFF紀錄編造思考。
OFF資料完整保留於raw source，沒有刪除；它們可做後續獨立OFF基線，未混入這次ON配方。

- 只訓練buyer；目前可用工具buy_listed／accept_offer／send_message／post_to_room／post_review／wait。
- 目標必須恰好一個工具、參數可解析且符合schema，原始工具結果ok=true；失敗／缺結果、swap等排除。
- 購買目標為qty1、尚未持有、當前board／offer真實存在、現金足夠且可見邊際收益為正。
  如果下回合紀錄明確揭露low且因此非正收益，也排除。沒有觀測到grade不臆造品質標籤。
- 排除round>5、最後一個行動再傳議價訊息、明確多件需求訊息，以及有可支付固定品質正收益卻wait的目標。
- 保留官方原本的reasoning，最多512tokens；整個assistant答案最多1024tokens，不截斷半個答案。
- 用原始observation與最多最近6turns重建prompt。超過12,288tokens時只移除最舊歷史，
  不截斷當前觀測／財務資料；system＋當前觀測仍太長則排除。
- past assistant reasoning依目前harness的工具歷史格式去除；失敗歷史與真實結果可以留在prompt，loss只監督最後答案。
- 原始Arrow／JSON中有schema與參數的null補欄，正規化時移除空欄，不改金額、庫存或效用。

篩選後1,347筆；每個排除原因與來源軌跡／訊息索引均有記錄，不用最終profit替模型洩漏未來資訊。

| Split | 決策樣本 | 原始軌跡 | 公開市場條件群組 |
|---|---:|---:|---:|
| Train | 890 | 191 | 43 |
| Validation | 201 | 37 | 6 |
| Test | 256 | 41 | 6 |

先以產品／list prices、共同budget、參與者與public overview做保守分組，再整組切分。
私人估值、persona或agent順序不同，不會把同一組公開市場拆成不同split；不是逐turn隨機切。
群組大小不同，所以樣本比例不會剛好80/10/10。Test只做CPU完整性核對，沒有模型在上面調參或選checkpoint。

訓練動作：432 buy_listed、70 accept_offer、172 send_message、20 post_to_room、129 post_review、67 wait。
這是curation後的官方資料分布，沒有硬塞4/1/3的合成配額或反覆補單一失敗題。
Train中458筆來自24商品市場、357筆12商品、75筆6商品；是多候選實況，已不是單一候選算術題。
Validation沒有post_to_room，只有1筆wait，因此不把它當作完整的房間／拒絕能力覆蓋；另有固定防呆檢查。
完整數字：docs/official_dataset_audit.json及training/data/buyer_official_v1/manifest.json。

## 第一輪固定配方

| 項目 | 設定 |
|---|---|
| 底座 | 原始Qwen/Qwen3.5-4B，revision851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a |
| 初始化 | 新LoRA，不接舊operations／repair權重 |
| LoRA | rank16、alpha32、dropout0、只改語言層、PEFT safetensors、無DoRA |
| Precision | bf16，gradient checkpointing，原生HF訓練 |
| Optimizer | AdamW，LR5e-5，weight_decay0.01，clip1 |
| Batch | microbatch1/GPU，兩卡，各accum4，global batch8 |
| 長度 | max sequence12288，完整assistant target≤1024 |
| Loss | 最後assistant完整completion CE；prompt／history不計loss，無額外tool-name加權 |
| 長度與schedule | 1完整epoch、112steps、6step warmup後cosine |
| Checkpoints | step28／56／84／112，保留adapter＋optimizer／scheduler／各rank RNG |
| 資料順序 | 固定seed20261120，全epochshuffle；890筆全涵蓋，最後batch補重播6筆，合計896筆曝光 |

這是預先固定、可重現的第一個基準配方，不是已證明最佳LR／epoch／資料配比。
三份推論persona在training/persona_versions/buyer_official_v1，保持短且不含手寫長檢查表；
訓練prompt保留官方多樣persona，推論三persona只作角色策略差異，後續比較時固定。

## 每個checkpoint看什麼

在任何optimizer更新前先測step0（原始底座，新增LoRA的B為零）。
step0／28／56／84／112各自：

1. 48筆官方validation，分tool報completion NLL。
2. 9筆真正未見過的官方validation自由生成，來自5類工具；不能要求所有合法替代行動都與teacher完全同字。
3. 6筆官方訓練集自由生成，每類工具1筆，選自首28步的batch prefix；step0尚未學，step28起確實全部見過。
4. 16筆固定當前規則防呆（8組cash正反例），涵蓋board／offer、等額／差一分／明顯不足／零現金、三persona。
   這些是CPU產生的獨立development probes，**不加入官方訓練資料**，也不是未見的正式評分測試。

每次共31次HF原生生成，使用真實qwen3_coder工具parser，不修復輸出、不塞gold reasoning。
Greedy、最多1024new tokens，方便固定比較；正式評分仍用官方temperature0.7／top_p0.95與8192tokens/round。
工具正常、現金條件、當前ID、捕捉正例與原始思考／輸出都保存。它們不是市場surplus。

若正例捕捉或有效動作相較step0下降至少4次，自動停止擴大損害並保存現有checkpoint。
這是大幅退化停損，不是要求小樣本每項都必須上升；低loss不能抵銷行為崩潰。
若挑development candidate，至少保住pair正確、現金／無工具錯誤不增加、正例至多少1；
再比較官方自由生成與validation NLL，並保留中間版本。沒有合格候選就明寫None，不默認最後一版最好。

## 目前直接執行

資料已在本機完成，不必重新下載／生成：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_official_buyer.py --gpus 0,1
```

這個命令由使用者啟動兩GPU。粗估含驗證先預留3～5小時，僅依前輪吞吐與這批長度外推；
原生HF生成、GPU共享狀態與fallback kernels可能改變耗時，以每階段實際progress／ETA為準。
開頭先載入、最長sequence與最長target前後向預檢，接著step0驗證；此時尚未開始optimizer更新。
訓練與生成各有完成數／總數、速度、elapsed、ETA，載入不可估時顯示estimating。
不啟動vLLM／完整市場、不替換提交adapter。較長官方prompt的實際顯存與速度仍待GPU預檢，CPU測試不冒稱GPU已驗證。

輸出`training/runs/<timestamp>_buyer_official_*`：

- `training_review.md`：先讀；底座與各checkpoint的NLL、正例／cash pair、現金錯誤、官方heldout及訓練題表現。
- `finished.json`、`checkpoint_scores.json`：完成／停損狀態、development candidate。
- `native_step_*.json`：每題自由生成的完整輸出與錯誤。
- `runner.log`、`metrics_rank*.jsonl`、`samples_rank*.jsonl`、`batch_plan.json`：進度、樣本曝光、loss、GPU記錄。
- `checkpoints/step_*`：中間權重及完整續跑狀態；`personas/`、`agent.yaml`為相應推論模式。

中斷後只用完整checkpoint續跑：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_official_buyer.py \
  --resume 實際run目錄/checkpoints/step_XXXX --gpus 0,1
```

必須是checkpoint_complete存在的存檔；完整恢復optimizer／scheduler／rank RNG，不是重開warmup。
資料、來源程式、LR／schedule／LoRA設定或persona／測試案例改了會拒絕exact resume。

## 第一輪之後的固定順序

先讀訓練前／後同條件的三類證據，確定是示範沒學穩、泛化不足，還是當前規則適應問題。
挑定checkpoint後才使用封存test做一次獨立確認；不以test反覆選模。
之後再恢復完整市場surplus比較，市場wrapper目前綁舊candidate，必須改成這次選出的官方基線再跑，不能誤用舊版分數。

只有基線成立後才做一個有依據的改動，例如補目前規則資料或品質策略；固定其餘條件與baseline做對照。
不把官方資料、更多steps、thinking ON或低loss當成高分保證。若結果不好，保留基線和失敗證據，先定位再改。

## 從零重現與CPU檢查

```bash
PYTHON_BIN=/home/hank/.cache/adl_final_venv/bin/python bash training/download.sh
/home/hank/.cache/adl_final_venv/bin/python -u training/official_buyer_data.py
/home/hank/.cache/adl_final_venv/bin/python -u training/train_official_buyer.py --cpu-check
/home/hank/.cache/adl_final_venv/bin/python -m unittest training.test_official_buyer
```

資料generator拒絕覆蓋，重建必須用新output並修改yaml的data_dir。
開發防呆題由CPU直接按固定seed生成，不依賴舊合成訓練資料目錄，不需要TA checkpoint或私人憑證。
模型與套件版本、硬體／時間會寫入本輪記錄；尚無本輪GPU訓練時間／成效實測。
CPU證據：docs/official_buyer_cpu_review.json。Codex協助資料與訓練設計，report應揭露LLM使用。
