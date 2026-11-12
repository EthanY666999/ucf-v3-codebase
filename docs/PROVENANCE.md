# 运行协议与来源

2B 源码取自 `EthanY666999/ucf-agent-rl` 的 `experiments/ucf-v3-no-sft`，固定提交 `1bc584b360db4b485d5e9b2065d3f9c82622e8f0`。保留原轨迹与反事实的采样、按题预算校准、HT 行加权、原位置 logits 捕获、网格提示和推理内核切换。独立发布去掉了原服务器的队列与租约入口，新增路径配置、显式设备选择、WebShop 协议和消融入口。此仓库实现的是 V3 首围栏分叉，整回合重掷的 V3.1 草案留在研究记录中。[实跑源码](https://github.com/EthanY666999/ucf-agent-rl/tree/1bc584b360db4b485d5e9b2065d3f9c82622e8f0/code) · [运行配置快照](../protocols)

`protocols/` 保留普通 FrozenLake 与 Sokoban 4,096 预算的四份初版配置、四份 scoped 内核配置，以及 Sokoban 10,240 预算配对的两份配置。只替换原主机绝对路径。包含配置不代表每一个组合都已完成正式训练。新运行通过 `configs/common.json`、规模/环境 JSON、`methods.json` 组合出 `launch.json`。原项目最早的短预算配置沿用前缀重算；本发布默认使用后来的实际采样 logits，避免混报。

网格数据含固定训练、验证和测试 ID。普通 FrozenLake 固定种子和生成器，Sokoban 及 FLH 另附冻结地图。哈希在 `data/SHA256.json`，每次启动重新核验。Sokoban 和 FLH 两个文件还含 500 条独立补充题，默认正式评估使用主测试集。rLLM 子集保留原 Apache 2.0 许可证，两套官方模型元数据各附上游许可证。[rLLM 源码](https://github.com/EthanY666999/ucf-agent-rl/tree/1bc584b360db4b485d5e9b2065d3f9c82622e8f0/code/vendor/rllm_source) · [冻结数据](https://github.com/EthanY666999/ucf-agent-rl/tree/4ab22056f471c79098e05bf41c11350f632c60ce/data)

## 复跑第 16 步配对

默认命令从原始模型开始。要重建历史 25/50 两个验证点，需要另行取得**原始**第 16 步完整 checkpoint，SHA256 为 `a4a4ccaae75a3f9056d17b924d4184facaed7b4ae677709ddb04a0daf1338c0b`。它已有 16 步无 SFT 的 UCF 更新。下列两臂分别载入相同模型、优化器和 RNG，参考模型仍为原始模型，先复评第 16 步，再更新到目标步。文件路径和指纹会写入运行配置。公共仓库不分发该存档。[历史起点与配对记录](https://github.com/EthanY666999/ucf-agent-rl/tree/1bc584b360db4b485d5e9b2065d3f9c82622e8f0/results/budget10240_paired25)

```bash
export MODEL_DIR=/mnt/models/Qwen3.5-2B
python run.py --size 2b --env sokoban --method ucf --anchor16 /mnt/archive/step16.pt --until 25
python run.py --size 2b --env sokoban --method grpo4 --anchor16 /mnt/archive/step16.pt --until 25
```

普通独立实验从零步运行不需要这份存档。使用另一个 `--output-root` 即可和历史起点的输出分开。
