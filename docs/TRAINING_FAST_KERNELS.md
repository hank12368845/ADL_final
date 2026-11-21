# HF訓練加速套件的隔離安裝

2026-11-20，本機CPU審查：Python3.11、Torch2.11.0+cu130、CUDA runtime13.0、Triton3.6.0、Transformers5.22.1、vLLM0.25.1。
準備階段causal-conv1d與FLA尚未安裝；使用者之後已完成隔離安裝，最新狀態見下節。系統nvcc是12.9，不能直接當成相符的CUDA13編譯器。
先前訓練log證明有慢速fallback；同時有長history、activation CPU搬運及checkpoint重算。沒有profiling，不能說整段耗時只由缺套件造成。

使用者允許提供安裝指令，現已準備CPU安裝腳本，**本次Codex沒有實際下載／安裝套件或執行GPU測試**：

```bash
/home/hank/.cache/adl_final_venv/bin/python -u training/install_fast_kernels.py --install
```

不加--install只顯示本機相容性與安裝計畫，不連網、不安裝、不查GPU。
本次市場使用原vLLM環境，可以先跑市場；不必等這個HF訓練加速安裝。

## 安裝內容

- 專案`.cache/fast_kernels/site-packages`：固定fla-core0.6.0、flash-linear-attention0.6.0、opt-einsum3.4.0。--no-deps保留現有Torch／Triton／Transformers／vLLM。
- 優先下載官方causal-conv1d1.7.0、與本機CUDA major／Torch minor／C++ ABI／Python／平台精確相符的wheel。
- 查閱的v1.7.0 release清單未見Torch2.11/cu13 wheel；若URL回404，預期走來源編譯。其他HTTP錯誤會停止，不冒充wheel不存在。
- 來源編譯時，在`.cache/fast_kernels/cuda13`安裝NVIDIA nvcc／CRT／NVVM13.0.88、runtime13.0.96、CCCL13.0.85，核對nvcc major/minor後才編譯。
- 取得官方main的確切commit，以該commit tarball建置；記錄commit與下載SHA，不把未凍結的main稱為可重現來源。
- MAX_JOBS2、CUDA_VISIBLE_DEVICES空字串、指定8.9架構，不自動偵測或占用GPU。不改系統CUDA或共用venv。
- 使用獨立pip zipapp，因目前venv沒有pip而snap uv在Codex sandbox不能執行；不為bootstrap改掉venv套件。

所有安裝、下載、編譯均有終端狀態與log；無法估算編譯剩餘時間時明示估算中，不虛構百分比。
安裝前要求至少3GiB可用空間，cache與暫存置於本專案prefix。
若已完成，程式拒絕覆寫prefix；部分失敗會保留log。要重試到另一組乾淨目錄可加--prefix /tmp/新的目錄。

## 完成後如何使用與驗證

安裝完成輸出`.cache/fast_kernels/installed.json`及env.sh；狀態為**installed_not_gpu_verified**，代表完成放置／建置，並不保證CUDA算子已驗證或已有速度提升。
receipt記載原環境版本、安裝包版本、來源commit、下載SHA與階段log。核心環境signature前後必須一致。
未來需要HF訓練時才啟用：

```bash
source .cache/fast_kernels/env.sh
```

此設定只增加目前shell的PYTHONPATH；新終端不啟用overlay即可回到原環境。本次市場命令不需要source它。
之後的HF GPU預檢仍應確認Qwen實際選到fast kernel、forward／backward正常、輸出／梯度容差與吞吐改善，再考慮調整CPU offload／MLP分塊。
不得因安裝成功就刪掉已驗證的顯存防護，也不宣稱市場推論自動變快。
目前CPU測試驗證安裝計畫與「精確wheel／404來源編譯」兩條流程的控制邏輯，模擬下載與pip，**不是實際安裝證據**。

## 使用者已完成安裝：CPU核對結果

`.cache/fast_kernels/installed.json`確認FLA/core0.6.0與causal_conv1d1.7.0安裝完成，核心Torch／vLLM等signature不變。
causal從官方commit `cd81f0413cad2fc1e6f17e785ac39f59aae690cd`，使用隔離CUDA13工具鏈成功建置，見03_causal_build.log。
Codex僅在CUDA_VISIBLE_DEVICES空字串下做CPU import，fla、fla.ops.gated_delta_rule、causal_conv1d、causal_conv1d_cuda皆成功；未執行GPU工作。
Codex sandbox不能在預設~/.tilelang寫快取，CPU驗證增加以下環境設定後成功；使用者正常shell不一定有此限制：

```bash
export TILELANG_CACHE_DIR=/mnt/data1/hank/ADL/2026-ADL-Final-Release/.cache/fast_kernels/tilelang-cache
```

證據存於results/basic_checks/20261120-220651_buyer_pair_dw5jjhar/kernel_cpu_import_review.json。
GPU不可見時FLA顯示CPU平台警告符合本次檢查設定，不能拿它判斷GPU部署會fallback。
狀態仍為installed_not_gpu_verified：還未驗證Qwen實際dispatch、GPU數值／梯度或速度，不因import成功就宣稱訓練已加速。本次兩場vLLM市場均未使用overlay。

## 官方參考

- [FLA安裝與後端說明](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)
- [FLA0.6.0版本與檔案](https://pypi.org/project/flash-linear-attention/0.6.0/)
- [causal-conv1d官方建置程式](https://github.com/Dao-AILab/causal-conv1d/blob/main/setup.py)
- [causal-conv1d v1.7.0 release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0)
- [NVIDIA CUDA pip套件與CUDA13命名](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pip-wheels)
- [NVIDIA nvcc13.0.88套件](https://pypi.org/project/nvidia-cuda-nvcc/13.0.88/)
