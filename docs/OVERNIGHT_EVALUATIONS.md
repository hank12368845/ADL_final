# 夜間排隊：Seller對照後接Buyer完整economy

使用者要求Buyer完整評估也排GPU，希望睡醒有結果。GPU仍由使用者啟動，Codex只完成CPU準備與測試。
不保證睡醒完成：其他使用者何時釋出GPU未知，單卡時間也可能較久。保留原本副本模式，TP2暫停。

## 已準備的實驗

1. Seller使用results/basic_checks/20261121-005859_seller_personas_1f4wxek7，v1／v2各一場完整5×10／30turn，900行動。
2. Buyer使用results/basic_checks/20261121-000742_step10_economy_a2txwbqx，固定step10與原persona／Thinking ON，官方Phase2的3economies×5rotations，共15場6750行動。

兩份實驗來源／權重／persona指紋已CPU核對通過，接用現有資料夾，完成市場不重跑。總計7650行動，不是額外訓練。
Seller原本的單獨queue需先由使用者Ctrl+C停止，避免它和night controller重複啟動同一實驗。尚未開始的等待可直接接手；已完成arm／market保留，半場依原wrapper規則重播。

## 啟動

預先準備的controller資料夾：results/basic_checks/20261121-010913_overnight_mh34xxnf。

```bash
tmux new-session -s adl-night -c /mnt/data1/hank/ADL/2026-ADL-Final-Release \
  '/home/hank/.cache/adl_final_venv/bin/python -u training/overnight_evaluations.py --resume results/basic_checks/20261121-010913_overnight_mh34xxnf'
```

Ctrl+B後放開、再按D可detach，SSH斷線不會因而停止tmux內工作。仍在執行時用tmux attach -t adl-night查看。
另建新controller可直接執行training/overnight_evaluations.py --gpus 0,1；其預設同上述實驗。不要同時開第二個controller，程式有本地互斥鎖。

每一步都有獨立顯存等待：GPU0／1都可，一張先單卡、兩張足夠則雙卡；每卡14GiB副本預算＋2GiB餘裕。等待15秒輪詢、queue ETA unknown，開始後沿用市場／economy進度與ETA。
Controller嚴格先Seller、後Buyer，不同時搶卡。每項一般失敗記錄後仍繼續另一個獨立實驗，避免整晚空等；手動中斷（SIGINT／exit130）則停止整條流程，不自動再開Buyer。
Buyer新queue入口training/queue_buyer_economy.py會在顯存預檢競態時重排，並保留已完成的官方市場；其他程式／模型錯誤停止該項，不無限重試。

## 結果

- controller內overnight.log：兩項完整終端輸出。
- overnight_status.json：即時running／success／failed／interrupted與各結果位置。
- overnight_review.md：兩項結束後的總表。
- Seller原資料夾的persona_comparison.md；Buyer原資料夾的economy_review.md與run_economy子目錄的官方standings.md。

success只代表程序正常完成，策略是否有效仍看各persona真實profit／surplus與錯誤。沒有自動訓練、選模或部署。
5項CPU控制流程測試通過：真實凍結實驗只讀預檢、順序執行／一般錯誤後續項／resume跳成功項、中斷停止、Buyer prepare不查GPU、Buyer顯存競態重排。等待／程序啟動均mock，無真GPU工作。
證據docs/overnight_cpu_review.json與/tmp/adl_overnight_cpu_tests.log。
