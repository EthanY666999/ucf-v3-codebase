# 运行协议与来源

2B 源码取自 `EthanY666999/ucf-agent-rl` 的 `experiments/ucf-v3-no-sft`，固定提交 `1bc584b360db4b485d5e9b2065d3f9c82622e8f0`。保留原轨迹与反事实的采样、按题预算校准、HT 行加权、原位置 logits 捕获、网格提示和推理内核切换。独立发布去掉了原服务器的队列与租约入口，新增路径配置、显式设备选择、WebShop 协议和消融入口。此仓库实现的是 V3 首围栏分叉，整回合重掷的 V3.1 草案留在研究记录中。[实跑源码](https://github.com/EthanY666999/ucf-agent-rl/tree/1bc584b360db4b485d5e9b2065d3f9c82622e8f0/code) · [运行配置快照](../protocols)

`protocols/` 保存 Sokoban 的六份历史配置，分别是 4,096 预算初版与 scoped 内核各两份，以及 10,240 预算的配对两份。新运行通过 `configs/common.json`、规模/环境 JSON、`methods.json` 组合出 `launch.json`。原版 FrozenLake 的基座评估 F01 和无 SFT GRPO4 实验 F02 可在原项目查阅。[F01 与 F02 的评估归档](https://github.com/EthanY666999/ucf-agent-rl/blob/4ab22056f471c79098e05bf41c11350f632c60ce/data/evidence/summary.json#L940)

Sokoban 和 FLH 各档都附有冻结地图、训练 10,000 题、验证 100 题、主测试 100 题及补充 500 题。2B 和 4B 按档共用同一批地图，启动时核验 SHA256。FLH 中档沿用难档每题的边长与冰面概率，以独立种子生成最短解 5–9 步的地图，并按地图哈希排除跨档和跨划分重复。可用 `python scripts/freeze_flh_medium.py --output /tmp/flh-medium` 重建，与 `data/SHA256.json` 核对。rLLM 子集和两套模型元数据各附上游许可证。[冻结数据与校验和](../data/SHA256.json) · [原项目数据](https://github.com/EthanY666999/ucf-agent-rl/tree/4ab22056f471c79098e05bf41c11350f632c60ce/data)

## 复跑第 16 步配对

默认命令从原始模型开始。要重建历史 25/50 两个验证点，需要另行取得**原始**第 16 步完整 checkpoint，SHA256 为 `a4a4ccaae75a3f9056d17b924d4184facaed7b4ae677709ddb04a0daf1338c0b`。它已有 16 步无 SFT 的 UCF 更新。下列两臂分别载入相同模型、优化器和 RNG，参考模型仍为原始模型，先复评第 16 步，再更新到目标步。文件路径和指纹会写入运行配置。公共仓库不分发该存档。[历史起点与配对记录](https://github.com/EthanY666999/ucf-agent-rl/tree/1bc584b360db4b485d5e9b2065d3f9c82622e8f0/results/budget10240_paired25)

```bash
export MODEL_DIR=/mnt/models/Qwen3.5-2B
python run.py --size 2b --env sokoban --method ucf --anchor16 /mnt/archive/step16.pt --until 25
python run.py --size 2b --env sokoban --method grpo4 --anchor16 /mnt/archive/step16.pt --until 25
```

普通独立实验从零步运行不需要这份存档。使用另一个 `--output-root` 即可和历史起点的输出分开。
