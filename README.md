# UCF V3.1 Code Base

本仓库负责从克隆到连续跑完 **Qwen3.5-2B、Sokoban、单卡**的默认队列。每项从相同的原始模型独立开始，无任务 SFT。换环境、换模型尺寸、使用多卡训练时，由使用方调整配置并在目标机器验收。

## 从克隆到连续队列

在具备足够显存的 Linux NVIDIA 训练机上准备 Python 3.12、匹配的 CUDA 驱动、官方模型目录和持久数据盘。下例仅需将模型和输出路径换成本机路径。[依赖、固定模型 revision 和硬件容量](docs/INSTALL.md)

```bash
git clone https://github.com/EthanY666999/ucf-v3.1-codebase.git
cd ucf-v3.1-codebase
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gpu.txt
python -m pytest -q

export CUDA_VISIBLE_DEVICES=0
export MODEL_DIR=/mnt/models/Qwen3.5-2B
export RUN_ROOT=/mnt/experiments/ucf-v3.1
python run.py --size 2b --env sokoban --method ucf --model-dir "$MODEL_DIR" --check
python run.py --size 2b --env sokoban --method ucf --model-dir "$MODEL_DIR" --smoke
python run_queue.py --size 2b --env sokoban --model-dir "$MODEL_DIR" --run-root "$RUN_ROOT"
python collect_results.py --run-root "$RUN_ROOT"
```

`--check` 核对冻结任务和分词器指纹。`--smoke` 在独立的 `smoke/` 目录执行一个真实 GPU 更新和验证。仅在 CPU 上验收时，安装 `requirements-cpu.txt`，运行 pytest，再用 `--model-dir metadata/Qwen3.5-2B --check` 校验随库元数据。

2B Sokoban 默认队列先跑 `ucf`（每条轨迹最多生成 **10,240 token**），再跑 `ucf-20480`（**20,480 token**）。两项都从同一个原始模型独立训练 400 步，使用 seed 42 和相同的 UCF V3.1 参数，各自保存检查点和评估结果。随后依次运行 `grpo4 → gigpo → ucf-merge → ucf-success → ucf-m1 → ucf-uniform → no-part-b → grpo8`。

可在 `configs/queue.json` 为每项设置 `seed` 和绝对步数 `until`（默认 42 和 400），并用 `size`、`env` 指定适用的模型规模和环境。重跑同一命令会跳过达标项，从已有存档恢复未完成项。失败重试一次，仍失败则记下原因并继续。进度、最近固定验证与失败记录在 `RUN_ROOT/queue_status.json`，汇总表为 `RUN_ROOT/results.md` 和 `RUN_ROOT/results.csv`。[采集、训练目标与指标](docs/EXPERIMENTS.md)

## 使用方调整的位置

1. **模型路径。** 用 `--model-dir` 指向包含权重、分词器及聊天模板的官方目录。换尺寸先核对 `configs/models.json` 中对应的 revision。
2. **环境和预算。** 在 `configs/<size>/<env>.json` 设置数据、地图、回合上限和整轨迹 token 预算。仓库提供 2B Sokoban / FrozenLake 和 4B Sokoban / FrozenLake / WebShop 预设。WebShop 另需准备模拟器、商品索引和冻结任务划分。[WebShop 搭建](docs/WEBSHOP.md)
3. **设备与多卡。** 在同一配置文件调整 `actor_device`、`micro_batch_size`、`micro_batch_token_limit`、`vllm_max_sequences` 和 `vllm_memory_fraction`。2B 默认将训练、参考模型和推理放在逻辑卡 0；现有 4B 示例将 actor 放在逻辑卡 1。数值内核切换与训练入口在 `runtime/`，模型放置及权重同步在 `code/train.py`。换布局后在目标机测峰值和吞吐。[设备配置](docs/INSTALL.md)

## 历史结果（V3，见标签 v3-final）

无 SFT 时，在动作 token 处换成其他后果的概率中位数只有 **0.0003**。因此 V3.1 将分叉前移到回合起点，重新采样推理和动作。历史存档和实现请使用 [v3-final](https://github.com/EthanY666999/ucf-v3.1-codebase/tree/v3-final)。

2B Sokoban 的 seed42 配对从同一份 **第 16 步 UCF 强化学习存档**分出，包含模型、AdamW 和随机数状态。该存档此前用 4,096 token 预算训练，两臂从第 17 步起使用 10,240。这组结果不能代替从零步独立启动的对照。

| 固定验证，100 题 × 4 | 第 25 步 | 第 50 步 | 第 50 步 pass@4 |
| --- | ---: | ---: | ---: |
| UCF V3 | 15.75% | 54.00%（216/400）| 80/100 |
| GRPO4 | 22.00% | 63.25%（253/400）| 83/100 |

第 17–50 步，UCF 生成 6,990 万 token，GRPO4 生成 5,494 万 token。时间和采样量都应随成功率一起报告。此表来自原实验的同题验证；4B 和 WebShop 的结果待测。[原项目配对报告与逐题统计](https://github.com/EthanY666999/ucf-agent-rl/tree/1bc584b360db4b485d5e9b2065d3f9c82622e8f0/results/budget10240_paired50)
