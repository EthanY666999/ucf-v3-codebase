# 训练环境与容量

## 模型

使用官方 Qwen3.5 原始目录，包含训练权重、tokenizer、config 和 chat template。两种规模使用相同 tokenizer 词表，但模板默认行为不同；项目显式使用 `enable_thinking=False` 并保留历史采样 token。空 think 块与模型自行生成的内容沿用生产约定。CPU 测试使用随仓库附带的两份元数据。权重可在外网机器按 revision 下载，然后整体移入训练服务器。[模型 revision 与校验和](../configs/models.json)

```bash
hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir /mnt/models/Qwen3.5-2B
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir /mnt/models/Qwen3.5-4B
```

## GPU 软件

运行主机使用 Linux x86_64、Python 3.12、具备 BF16 支持的 NVIDIA GPU。2B 生产环境为 PyTorch 2.10.0（CUDA 12.8 构建）、vLLM 0.19.1、Transformers 5.22.2。优先使用同版本 CUDA 镜像或内网 wheelhouse，确保 vLLM 的二进制扩展与 PyTorch ABI 和驱动相容。CPU 测试可在另一环境执行。[vLLM 安装说明](https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/) · [PyTorch 安装版本表](https://pytorch.org/get-started/previous-versions/)

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r requirements-gpu.txt
python -m pip check
nvidia-smi
python -c "import torch,vllm,transformers; print(torch.__version__,torch.version.cuda,vllm.__version__,transformers.__version__,torch.cuda.device_count())"
python -m pytest -q tests/test_original_logits.py
```

使用内网 wheelhouse 时可将安装行替换为 `python -m pip install --no-index --find-links /mnt/wheels -r requirements-gpu.txt`。`VLLM_BATCH_INVARIANT` 只在推理阶段生效，优化阶段恢复原生 PyTorch 算子。UCF 主分支使用原始采样时的完整 logits 观察器与 Python 补集采样。部署验收应覆盖一次完整采样、反向传播、权重同步和验证。

## 设备、磁盘和恢复

2B 配置默认同卡。4B 配置默认两张可见卡，在单进程内显式把 FP32 actor 和 AdamW 放到逻辑 `cuda:1`，参考模型和 vLLM 在 `cuda:0`。权重逐张量复制并检查同步。`CUDA_VISIBLE_DEVICES` 设置物理卡顺序，`--actor-device` 可指定逻辑卡。一次启动一个训练进程，不用 torchrun/DDP。跨卡实现需在目标机验收。相同 run 目录有互斥文件锁；在不同目录安排作业时由调度器分配卡。

4B 文本模型有 4,205,751,296 个参数。actor 参数和梯度各约 15.67 GiB，普通 AdamW 的两个状态约 31.34 GiB，训练卡静态合计约 **62.67 GiB**。参考模型与推理文本权重各约 **7.83 GiB**。这些是静态下界，真实峰值还包含激活、KV、完整词表向量和同步工作区。单 80G 全部同放的静态下界约 78.34 GiB。两张 80G 的默认路径使用单轨迹训练微批，推理并发 16；仍须实测一条接近预算上限的轨迹，普通一轮冒烟未必触及峰值。[原 4B 参数计数](https://github.com/EthanY666999/ucf-agent-rl/blob/4ab22056f471c79098e05bf41c11350f632c60ce/docs/MEMORY_AND_RUNTIME.md)

可调节 `--micro-batch-size`、`--micro-batch-token-limit`、`--vllm-max-sequences` 和 `--vllm-memory-fraction`。完整长序列超过微批 soft limit 时仍单独训练，改变整条轨迹的 token 上限属于不同实验。主机内存也需容纳 observer 的候选向量和 checkpoint 载入；每份向量约 `4 × vocab_size` 字节。

4B 模型与优化器单份存档的静态估计约 47 GiB，2B 约 21 GiB。原子替换期间新旧两份会并存，还要给轨迹和验证记录留空间。用 `RUN_ROOT` 指向持久数据盘，替换前程序检验空余量。中断后用同配置和同一输出根运行 `--resume`，未提交到存档的后续步证据保留在恢复目录。权重保留在服务器，日常对外分享轻量收据。
