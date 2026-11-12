# 运行协议与来源

当前版本使用 UCF V3.1 整回合重掷。已发表的 2B Sokoban V3 配对结果对应 `v3-final`，原始 2B 实验源码来自 `EthanY666999/ucf-agent-rl` 的提交 `1bc584b360db4b485d5e9b2065d3f9c82622e8f0`。[历史版本](https://github.com/EthanY666999/ucf-v3.1-codebase/tree/v3-final) · [原始配对记录](https://github.com/EthanY666999/ucf-agent-rl/tree/1bc584b360db4b485d5e9b2065d3f9c82622e8f0/results/budget10240_paired50)

六份历史配置存放在 `docs/archive/`，仅作记录。需要恢复历史 V3 存档时，请在独立 checkout 切换到 `v3-final`，按该标签的 README 操作。新实验通过 `configs/common.json`、规模/环境配置和 `configs/methods.json` 合成自己的 `launch.json`。[历史配置](archive/)

Sokoban 和 FLH 各档都附有冻结地图、训练 10,000 题、验证 100 题、主测试 100 题及补充 500 题。2B 和 4B 按档共用同一批地图，启动时核验 SHA256。FLH 中档沿用难档每题的边长与冰面概率，以独立种子生成最短解 5–9 步的地图，并按地图哈希排除跨档和跨划分重复。可用 `python scripts/freeze_flh_medium.py --output /tmp/flh-medium` 重建，与 `data/SHA256.json` 核对。rLLM 子集和两套模型元数据各附上游许可证。[冻结数据与校验和](../data/SHA256.json) · [原项目数据](https://github.com/EthanY666999/ucf-agent-rl/tree/4ab22056f471c79098e05bf41c11350f632c60ce/data)
