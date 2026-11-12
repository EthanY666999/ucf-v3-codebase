# 发布验收

2026 年 11 月 12 日使用 CPU 核对。仓库附的 CPU 测试使用两款官方 tokenizer，与模型权重分离。

| 环境 | 实际覆盖 |
| --- | --- |
| macOS，Python 3.12，Transformers 5.22.2、PyTorch 2.10.0 | 51 项通过。五种模型/环境、基线与消融的 CLI 配置展开，网格数据和地图哈希、真实模板和动作 token、补集采样、HT 行、CPU 张量上的优化与权重同步、断点恢复 |
| WebShop CPU fixture（包含于上面的 51 项） | HTTP 接口、完整指令块与可见操作、独立历史重放、购买 1.0 / 0.4 两种回报、GiGPO 状态分组、UCF 分叉及日志。模型生成和商品目录用小型 fixture |
| Linux，生产版本 vLLM 0.19.1、PyTorch 2.10.0，进程可见 GPU 数量为 0 | 单独执行 `tests/test_original_logits.py` 的 20 项。检查首围栏位置、完整概率向量、原始 token logprob、请求重排、异常与缓存释放。Mac 缺少 vLLM 的跳过模块在这里补齐 |

目标机仍须运行完整商品索引的 `scripts/check_webshop_live.py` 和各 4B 环境的 `--smoke`。CPU 上的小张量同步无法测量跨 A100 的实际峰值、吞吐或成功率。[WebShop 验收](WEBSHOP.md) · [训练机验收](INSTALL.md)
