# 发布验收

2026 年 11 月 12 日使用 CPU 核对。仓库附的 CPU 测试使用两款官方 tokenizer，与模型权重分离。

| 环境 | 实际覆盖 |
| --- | --- |
| macOS，Python 3.12，Transformers 5.22.2、PyTorch 2.10.0 | 65 项通过。覆盖五种模型与环境组合、两档 FLH 的数据和配置、基线与消融、分词与动作、补集采样、CPU 权重同步和断点恢复。回归测试用模拟采样和优化，在难档成功率与 pass@4 始终为 0% 时走完 400 步，核验了每 25 步的验证、最终测试和地图哈希。1 个 vLLM 测试模块因本机缺少 vLLM 跳过 |
| WebShop CPU fixture（包含于上面的 65 项） | HTTP 接口、完整指令块与可见操作、独立历史重放、购买 1.0 / 0.4 两种回报、GiGPO 状态分组、UCF 分叉及日志。模型生成和商品目录用小型 fixture |
| 上一版 Linux 验收，vLLM 0.19.1、PyTorch 2.10.0，进程可见 GPU 数量为 0 | 单独执行 `tests/test_original_logits.py` 的 20 项。检查首围栏位置、完整概率向量、原始 token logprob、请求重排、异常与缓存释放 |

另在 CPU 上逐图核验了两档 FLH 共 21,400 张地图的 SHA256、边长和 BFS 最短路，回放每条最短路都到达终点。两档及各划分之间无重复地图。2B 和 4B 在 20,480 token 预算下的第 0 步成功率、pass@4 和耗时由训练机用真实权重测量。[第 0 步验证](EXPERIMENTS.md#frozenlake-第-0-步验证)

目标机仍须运行完整商品索引的 `scripts/check_webshop_live.py` 和各 4B 环境的 `--smoke`。CPU 上的小张量同步无法测量跨 A100 的实际峰值、吞吐或成功率。[WebShop 验收](WEBSHOP.md) · [训练机验收](INSTALL.md)
