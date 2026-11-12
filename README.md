# UCF V3 · Qwen3.5 智能体强化学习

从原始 Qwen3.5 出发，用相同的任务、提示、回合上限和验证集比较 **UCF V3、GRPO 和 GiGPO**。所有发布配置都跳过任务 SFT。运行入口是 `run.py`，Linux 内网机器可以使用预先准备好的模型目录，训练期间无需访问模型 API。

| 模型 | 环境参数 | 每局回合 | 整条轨迹的回复 token 上限 | 证据 |
| --- | --- | ---: | ---: | --- |
| 2B | `sokoban` | 10 | 10,240 | 有原项目同起点 UCF / GRPO 配对训练记录 |
| 2B | `frozenlake` | 10 | 4,096 | 原版地图与 V3 计划配置已归档，本仓库覆盖 CPU 环境验收 |
| 4B | `sokoban` | 10 | 10,240 | 同协议移植 |
| 4B | `flh` | 20 | 10,240 | 难版 FrozenLake，冻结 9–12 边长、最短解 7–16 步的地图 |
| 4B | `webshop` | 15 | 10,240 | 官方文本模拟器适配，自由搜索词和可见按钮；需要单独准备商品目录 |

4B 三组均需在合作者机器上做首轮 GPU 冒烟。仓库附网格任务、地图校验和及两款官方 tokenizer 的固定版本。模型权重和 WebShop 商品索引由运行方提供。[安装与硬件](docs/INSTALL.md) · [WebShop 搭建与验收](docs/WEBSHOP.md)

## 从克隆到第一组实验

```bash
git clone https://github.com/EthanY666999/ucf-v3-codebase.git
cd ucf-v3-codebase
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pytest -q
python run.py --size 4b --env sokoban --method ucf --model-dir metadata/Qwen3.5-4B --check
```

在装好 GPU 依赖的训练环境里，把下面两个路径改成自己的。先跑一个真实更新，随后独立启动正式目录。4B 双卡示例把参考模型和 vLLM 放在可见卡 0，训练模型放在可见卡 1。

```bash
export MODEL_DIR=/mnt/models/Qwen3.5-4B
export RUN_ROOT=/mnt/experiments/ucf-v3
export CUDA_VISIBLE_DEVICES=0,1
python run.py --size 4b --env sokoban --method grpo4 --smoke
python run.py --size 4b --env sokoban --method grpo4 --until 25
python run.py --size 4b --env sokoban --method grpo4 --resume --until 50
```

`--print-config` 输出完整训练参数，无需显卡。`--check` 校验冻结数据和模型元数据。`--smoke` 在独立目录运行 1 次参数更新，用验证题检查前后结果。`--until` 是绝对训练步数。正式任务每 25 步在固定验证集上每题采样 4 次，完整 400 步后评估测试集。原始训练命令、归档配置和从第 16 步复跑的条件见[版本与来源](docs/PROVENANCE.md)。

## 合作者运行顺序

每个环境先跑 `grpo4`，再跑 `ucf` 和 `gigpo`，三者分别从同一个原始模型起步。4B 依次做 Sokoban、难版 FrozenLake 和 WebShop。2B 做 Sokoban 和原版 FrozenLake。先看 seed42 的 25、50 步固定验证，再用 seed43 重复需要报告的配对。`grpo8` 检验把额外采样预算用于独立轨迹的效果。另附 StarPO-S 的组过滤组件对照和两项可执行机制消融。[基线、消融与交回结果](docs/EXPERIMENTS.md)

## 当前能引用的结果

2B Sokoban 的 seed42 配对从同一份 **第 16 步 UCF 强化学习存档**分出，包含模型、AdamW 和随机数状态。该存档此前用 4,096 token 预算训练，两臂从第 17 步起使用 10,240。这组结果不能代替从零步独立启动的对照。

| 固定验证，100 题 × 4 | 第 25 步 | 第 50 步 | 第 50 步 pass@4 |
| --- | ---: | ---: | ---: |
| UCF V3 | 15.75% | 54.00%（216/400）| 80/100 |
| GRPO4 | 22.00% | 63.25%（253/400）| 83/100 |

第 17–50 步，UCF 生成 6,990 万 token，GRPO4 生成 5,494 万 token。时间和采样量都应随成功率一起报告。此表来自原实验的同题验证；4B 和 WebShop 的结果待测。[原项目配对报告与逐题统计](https://github.com/EthanY666999/ucf-agent-rl/tree/1bc584b360db4b485d5e9b2065d3f9c82622e8f0/results/budget10240_paired50)
