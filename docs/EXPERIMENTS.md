# 基线、消融与交付

同一模型与环境的主对照共享原始权重、题目流、seed、整轨迹 token 上限、最大回合数、优化器、验证题和温度。UCF V3 在每题四条原始轨迹中寻找带首围栏的节点，用其他题的候选数量校准均匀选点概率，每题目标约两处分叉。原动作和反事实按后果分层，从原始采样位置的完整词表分布计算层概率，按完整补集采样；选中节点以实际 `1/ρ` 加权。Part B 使用同题其他三条的平均回报做基线，β=0.25，mask 从分叉 token 到该回合末尾。训练与评估在第一个完整动作块处按采样 token 边界截断。[运行实现](../code/ucf/ucf_v3.py) · [完整词表补集](../code/ucf/v3_sampling.py)

| `--method` | 做什么 | 用来回答什么 |
| --- | --- | --- |
| `grpo4` | 每题四条独立原始轨迹 | 同组大小的主基线 |
| `ucf` | GRPO4 加均匀选点与反事实 Part B | 分叉收益能否抵消其额外采样成本 |
| `gigpo` | 同框架 GiGPO，四条轨迹。轨迹优势加相同题和状态下的步级相对优势，γ=0.95，ω=1 | 对照另一种步级信用分配 |
| `starpo-s` | StarPO-S 组件对照：按组回报标准差保留最高 50%，KL 系数 0，clip 为 0.2/0.28 | 检验组过滤与稳定化设置；本地实现只覆盖列出的组件 |
| `grpo8` | 每题八条原始轨迹，验证仍采四次 | 用额外算力多采独立轨迹的收益。tokens 和小时需实测 |
| `no-part-b` | 执行和记录 UCF 分叉，β=0，更新只用原始行 | 隔离 Part B 的更新作用，保留分叉采样成本 |
| `token-partition` | 网格保留位置、HT 和 β，只把动作词映射成四个方向加剩余词类，不再合并同后果的动作 | 额外一步环境信息是否有价值。WebShop 默认已使用动作族分层 |

“均匀”指同题合格候选有相同的 ρ。完整 V3 移除了手工过程分 S；网格上仍有环境知识，例如 Sokoban 的角落死局标签、FrozenLake 的洞和终点。`token-partition` 就是检验这部分贡献的一组。WebShop 的 `SEARCH / CLICK / RESIDUAL` 属于首 token 类别，解析与首 token 不一致的比例会记录，不能将该分层解读为精确预测下一页。

先在同一台机器上比较各臂相同步数，然后按累计生成 token 和训练小时补充比较。GiGPO 官方 WebShop 配置使用不同模型和 group=8，本文的同框架 group=4 是匹配对照；单列官方发表值时请标明模型、奖励、划分、动作上限和评估采样。WebShop 同时报告成功率和官方平均 score。[官方 GiGPO WebShop 配置](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/examples/gigpo_trainer/run_webshop.sh) · [RAGEN / StarPO-S](https://github.com/mll-lab-nu/RAGEN) · [GiGPO 论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html)

## FrozenLake 第 0 步验证

2B 和 4B 各用自己的原始模型先测难档。两种规模共用同一份固定验证集，100 题、每题 4 条。每局 20 回合，整条轨迹最多生成 20,480 个回复 token。以下命令只评估验证集，完成后退出。训练机实际耗时见 `validation_0000.json` 的 `seconds`（仅采样与统计）。

```bash
python run.py --size 2b --env flh --method grpo4 --model-dir /mnt/models/Qwen3.5-2B --validate-base
python run.py --size 4b --env flh --method grpo4 --model-dir /mnt/models/Qwen3.5-4B --validate-base
```

结果在 `RUN_ROOT/base-validation/<规模>_flh_grpo4_seed42/`。单条成功率按 400 条轨迹计，pass@4 按 100 道题计。查看 `flh_calibration.json`，两个区间都包含端点。

| 第 0 步结果 | 后续安排 |
| --- | --- |
| 单条成功率 5%–20%，且 pass@4 为 20%–50% | 选用该档地图，开始同起点 GRPO4、UCF 和 GiGPO |
| 任一指标低于下限 | 难档退到中档，重新做第 0 步验证。直接启动正式难档训练时也会在第 1 次更新前检查并退出 |
| 两项均达到下限，至少一项超过上限 | 记录为 `above_target`，保留实测数值；此协议未预设更高难度 |

中档同样使用边长 9–12 的冻结地图，预设 BFS 最短解为 5–9 步，难档为 7–16 步。两档使用独立地图，沿用相同回合数和 token 预算。若中档仍低于目标，报告中会保留 `below_target`。选择难度只使用验证集；同一模型规模的各方法和 seed 都使用选定的同一档。

```bash
python run.py --size 2b --env flh --flh-difficulty medium --method grpo4 --model-dir /mnt/models/Qwen3.5-2B --validate-base
python run.py --size 2b --env flh --flh-difficulty medium --method grpo4 --model-dir /mnt/models/Qwen3.5-2B --until 25
```

如果希望 GRPO4 在难度合适时直接训练，可以直接使用 `--until 25`。它会先做同一项第 0 步验证，低于下限时退出，处在目标区间时进入训练，可省去对该运行单独测一遍。中档运行目录名包含 `flh_medium`，两档的记录分别保存。汇总不同规模时同时列出档位和地图哈希。

## 命令

```bash
python run.py --size 2b --env flh --method gigpo --until 25
python run.py --size 4b --env flh --method no-part-b --until 25
python run.py --size 4b --env sokoban --method token-partition --seed 43 --until 25
python run.py --size 4b --env webshop --method ucf --until 25
```

以上命令从环境变量 `MODEL_DIR` 读取对应**同规模**模型。每个方法都独立保存。第 25、50、100 步是看历程的验证点；使用同一批验证题分析差距，保留失败和截断。测试集只在事先选定的最终训练预算评估。四个样本来自同一题，区间按题配对重采样，训练 seed 单列。

## 结果

在 `RUN_ROOT/train/<规模>_<环境>_<方法>_seed<seed>/` 内保留 `launch.json`、`config.json`、`hardware.json`、`numerical_runtime.json`、`train_metrics.jsonl`、`validation_*.json[l]` 和最终测试文件。记录原始四条的全错/混合/全对比例，救回率分母为全部原始全错题，分叉数、候选覆盖、KL、生成 token、训练与验证用时分别汇总。UCF 的 `ucf_logs/budget_*.json` 与 `select_*.jsonl` 可回查具体节点。WebShop 另保存冻结目录的 `manifest.json`。论文表格用 `success_rate_pct`（每条轨迹）与 `pass_at_4_pct`（每题至少一次成功）两个分母。网格还有每条轨迹的步数和是否到达 token 上限。

内网可只交回汇总数字、配置、seed、源码提交、数据 SHA 和评估题数。完整记录留在运行主机。公开仓库默认忽略权重、存档及运行输出。
