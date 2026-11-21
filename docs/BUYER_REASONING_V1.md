# Buyer Reasoning v1：原始觀測、短計算、完整 epoch 續跑

使用者決定改用thinking ON，先確認有用再接1～2個完整epoch。這是一條新方法：
原生思考格式、簡短計算標籤與完整資料覆蓋一起調整，不能把結果單獨歸因為ON開關。
GPU仍由使用者啟動；Codex已完成CPU準備，不會自動開服務、訓練或等待GPU。

## 訓練與提交模式

新config為 `training/configs/buyer_reasoning_v1.yaml`，thinking=true。
資料、原生chat template、推論request與輸出agent.yaml都為ON。
每筆原始observation與歷史內容保留；只增加靜態persona的簡短計算指示，以及訓練答案的reasoning_content。
沒有修改TA的adlmarket/、工具、parser、行情或預算，也沒有用Python在推論前注入餘額。

官方模板產生 `<think>` → 簡短核對 → `</think>` → 一個原生工具呼叫。
例如：`C=20.50, P=20.51*1=20.51, C-P=-0.01; ... wait`。
核對另含持有量、庫存／offer有效性、數量與value-P；標籤只讀visible state，
與原引擎驗證過的工具答案逐筆比對，沒有把隱藏成本或答案加到輸入。
所有reasoning target最長限制192 tokens；這是資料檢查，不是推論時硬截斷思考。
推論時靠persona要求一兩句話，實際長度必須看評估。

buyer範例agent.yaml已設reasoning:true；其checkpoint仍是原starter identity LoRA，
不是這次尚未訓練的候選。舊OFF LoRA保留在原run；舊診斷仍依自身plan明確跑OFF。
新run另保存ON的agent.yaml與三個短persona，正式選模後需連同候選LoRA一起安裝，不能混用設定。

## 第一階段：32 steps

本機CPU資料已準備，直接跑：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_reasoning_v1.yaml --gpus 0,1
```

從原始Qwen3.5-4B底座新建rank16／alpha32 LoRA，LR1e-5、warmup4、batch8，thinking ON。
loss沿用completion CE＋1倍mean tool-name CE；完整completion包含簡短reasoning，
本輪不繼續掃首token權重。兩卡DDP，各microbatch1、累積4次。
每16步保存adapter、optimizer、scheduler與兩rank RNG；有step／micro進度、elapsed、速度及ETA。
啟動前檢查資料、模型快取、逐卡剩餘顯存與磁碟，兩卡不足就停止。

原v2.1實測約59秒／step，32步可先粗估30～40分鐘；ON目標長度與負載改變可能影響速度，
這不是實測保證，以實際ETA為準。輸出training/runs/<time>_buyer_operations_*/，
runner.log、metrics.jsonl、batch_plan、各rank sample trace、checkpoint與完成紀錄均保存。
Ctrl-C可停止整組worker；已完整保存的checkpoints可續跑，沒保存的microstep不會恢復。

資料為3072 train、384 validation、384 test，維持原市場split。
新sampler每batch兩個board pair＋兩個offer pair，動作2buy／2accept／4wait；
每epoch每筆資料恰好看一次，不把4/2/2重抽樣曝露量冒稱完整epoch。
32步為256筆不重複訓練資料（每卡128），八種family及四種cash variant均涵蓋。
全資料wait比例50%，可能仍有偏向等待的風險，所以正例召回是必查項。

## 第二階段：base ON 與 trained ON 同題評估

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_buyer_reasoning.py \
  PILOT_RUN --gpus 0,1
```

PILOT_RUN換成完成訓練的run目錄。預設checkpoint為step_0032，另可用--checkpoint final。
對照為原始底座ON與新候選ON，兩者使用相同persona及原始觀測，
不把OFF訓練LoRA切成ON當正式控制。兩個held-out validation市場，共64題，
三persona×兩seed×兩模型=768calls，雙卡各384，配對兩側及模型均同卡。
服務啟動與呼叫完成數、速度、elapsed及ETA在終端顯示；結果在results/basic_checks/，
若持久磁碟不足明示fallback到/tmp/adl_final_basic_checks。

先讀cash_decision_pairs.md與reasoning_gate.md，另存每次reasoning、usage、耗時、截斷與原生工具結果。
generated tokens = output_tokens＋reasoning_tokens，避免漏算thinking。
正式每輪8192 token額度完全保留；這是獨立狀態重播，不是完整六turn預算測試。

長跑gate是開發門檻，不是課程分數，條件全部需通過：

- offer兩側皆正確比base ON至少多4組，且至少25%。
- 買不起offer的交易錯誤減少，offer正例捕捉至少90%。
- 整體正例捕捉不比base少超過2筆，各persona的pair正確數不退化。
- 沒有截斷、token usage完整、至少90%回應可取得reasoning。
- 含thinking的單次輸出p95≤1024 tokens，作為六turn成本的初步篩檢。

gate通過不保證正式surplus或每輪絕不超額；這些仍需完整市場驗證。
若base ON已近滿分，沒有擴大cash專項訓練的明確收益，gate可能不通過，應先看市場表現。

## 第三階段：通過後接續1～2 epochs

3072/8=384 steps／epoch，兩epoch為768 steps。從pilot的step32接續，
一epoch額外352步；LR固定以768步為總horizon，避免pilot用完cosine後再重啟warmup。
同時恢復LoRA、optimizer、scheduler與每rank RNG，sample plan的舊prefix必須一致。
這是真正接續，不是僅載入adapter卻重設optimizer。訓練source／設定／資料漂移時拒收。

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/train_buyer_operations.py \
  --config training/configs/buyer_reasoning_v1.yaml --gpus 0,1 --epochs 1 \
  --resume PILOT_RUN/checkpoints/step_0032 \
  --gate-report EVAL_RUN/reasoning_gate.json
```

也可--epochs 2直接以768為目標；或一epoch完成後重新評估，
用其新gate與step_0384接續--epochs 2。續跑候選權重須與gate實際評過的權重一致。
一epoch約6～7小時、兩epoch約12～14小時僅按舊速度外推，尚無ON完整epoch實測。
長跑每64步保存並做32筆teacher-forced validation，不以loss自動宣告最佳checkpoint。
後續由使用者挑checkpoint做生成／市場收益評估；不自動部署至正式agents。

## CPU證據與重現

CPU檢查已包含原模板reasoning／tool順序及mask、3840筆visible target與原標籤一致、
觀測及歷史原文保留、完整兩epoch覆蓋與雙rank重建、pilot prefix一致，
tiny PEFT權重載入、optimizer／scheduler／RNG精確續跑、原生ON request、
雙卡768-call假worker、gate與token超額拒收。假worker高分不是實際模型效果。
詳細紀錄見docs/buyer_reasoning_cpu_review.json。

新機器先依training/README下載底座及套件，再依序：
buyer_operations_v2_1_data.py → buyer_reasoning_data.py → 本版訓練 → 評估 → 通過後續跑。
不需下載／訓練舊OFF LoRA或TA權重，沒有新增依賴、付費API或私有憑證。
本版程式由Codex協助，report需揭露LLM使用。提交只帶最終adapter_config.json與
adapter_model.safetensors、三persona、agent.yaml及訓練重現程式；不要夾帶optimizer、RNG或中間權重。
