# Step10官方Phase2 economy評估

使用者已要求先評估step10，不再訓練。GPU由使用者啟動：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_step10_economy.py --gpus 0,1
```

此啟動器真正執行官方`run_economy.py --phase 2 --config <隔離設定> --launch-vllm`。
固定surgical `20261120-203130_buyer_surgical_3hb_nseg/checkpoints/step_0010`、原三persona、Thinking ON。
不改agents內的部署checkpoint、不訓練、不改engine；預設新seed3。

## 評估範圍

- 官方configs/economy.yaml的Phase2規則、TA pool、評分anchors、casting、padding、3economies及5次seller庫存rotation均保留。
- 指定我們的buyer作唯一學生角色，避免glob混入其他實驗；每場5seller×10buyer、5round×6turn，15場共6750行動。每個學生persona450行動。
- TA sellers從公開pool抽樣，TA buyers作anchors與padding；因此與先前手選對手的單場組成不同，不能把raw變化直接當模型退化／進步。
- 三economies都沿用官方本地設定的同一public商品檔；正式private商品與更大的TA pool不在本次評估範圍。
- 官方溫度0.7、top_p0.95、context64000、每round8192輸出token與原工具、觀測、history均保留。

## 雙卡、進度與續跑

模型一次載入，官方runner依序跑完15場；每場內依TA生成成本及官方seat出現次數分流到兩個vLLM副本。學生三persona共用step10 adapter、固定server0。
每卡按剩餘顯存扣2GiB，最低14GiB啟動預算；只剩一張可用時明示單卡，不終止其他工作或宣稱能動態擴張。
路由使用inline TA ModelSpec，保留每個TA的ID、variant_of、persona與權重；CPU驗證全部15場座位／分組指紋一致，不能因分流而改變官方市場。

終端每5秒顯示0～6750總行動、0～15完成市場、當前市場0～450、round／turn、三persona各0～450、elapsed、速度與ETA；載入或資料不足時顯示估算中。
依既有單場15～23分鐘可先預留約4～6小時，對手組成與生成長度不同，這只是規劃，實際以終端ETA為準。
不使用HF訓練加速overlay；啟動子程序會移除該overlay的PYTHONPATH項目。

結果預設results/basic_checks/<time>_step10_economy_*；先檢查10GiB可用空間，不足則明示退回/tmp/adl_final_basic_checks。
固定buyer與TA路由副本放專案.cache/economy_eval/<run名>，避免官方agent resolver不接受repo外代理路徑。
runner.log與progress.log保留過程，官方run_economy_<time>/markets/<id>保存各場transcript、events、完整prompts、final及scores。

中斷後用實際wrapper資料夾續跑：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/evaluate_step10_economy.py \
  --resume results/basic_checks/實際_step10_economy_資料夾 --gpus 0,1
```

直接使用官方resume：保留有scores.json的完整市場，未完成場從頭重跑；不是從半場接續。重查資源與路由，座位／persona／權重仍須一致。
Mock與真模型不能混用resume。--prepare-only只做CPU設定與指紋驗證，不查GPU或啟服務；--mock僅CPU假模型流程測試，不能當成績。

## 跑完看什麼

啟動器會自動產生economy_review.md/json，不需要額外評估指令：

- 各persona15場平均surplus、最低／最高、每economy均值。
- 零邊際效用支出、錯掛牌、cash不足、截斷、額度wait、最長連續重試失效配對與逐題草稿。
- 逐單位重算utility／spent／cash／surplus核對官方scores，完整6750次行動及雙卡實際load。
- 官方standings.md/json、anchors及grade原樣留在run_economy_<time>。

本地舊版student列仍pool三persona；逐persona讀結果，不把它直接當FAQ最佳persona的正式最後分數。不能把normalized當正式grade，也不自行推定未公布的選優／正規化順序。
跨rotation結果相關，不把15場當15個完全獨立樣本。這次只有step10的官方規模結果，不能當成step84與step10已完成同條件economy對照。
沒有自動升級／部署／重訓／下一輪GPU工作，完成後使用者回報結果再分析。

## CPU流程驗證

`training/test_step10_economy.py`驗證15場官方casting、step10權重指定、雙端點保留分組／權重、第二卡忙碌時單卡fallback、完整6750行動mock與逐單位帳目、完成後不重跑、半場中斷保留14場及進度檔重建。
Mock與GPU查詢假資料只驗證控制流程，不代表任何真模型分數、GPU顯存或速度證據。真實GPU評估仍由使用者啟動。
