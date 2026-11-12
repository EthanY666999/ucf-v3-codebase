# 4B WebShop

训练容器使用本地 HTTP 连接到独立的 Princeton WebShop 官方文本模拟器。它处理自由搜索词、点击商品和选项、翻页和购买，所有实际交易都在模拟器内。传给策略的只有任务说明、当前文本页、搜索框状态与可见标签。固定步数和奖励规则作用于全部训练臂。[官方环境源码](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/web_agent_site/envs/web_agent_text_env.py)

## 1. 一次性准备模拟器

在新代码仓库根目录记下 `UCF_CODE`，将官方商品与索引放在有容量的磁盘上。已有内网 WebShop 安装可以直接使用同一 revision 和完整 human-goal 目录。官方安装步骤使用 Python 3.8.13、Java、spaCy 模型与 Lucene 索引，应独立于 GPU 训练环境。[官方安装说明](https://github.com/princeton-nlp/WebShop/tree/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd#-setup)

```bash
export UCF_CODE="$PWD"
git clone https://github.com/princeton-nlp/WebShop.git /mnt/benchmarks/WebShop
cd /mnt/benchmarks/WebShop
git checkout 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
conda create -n webshop-upstream python=3.8.13 -y
conda activate webshop-upstream
bash setup.sh -d all
PYTHONPATH="$PWD" python "$UCF_CODE/scripts/webshop_server.py" --port 8765
```

## 2. 在训练终端验收

以下命令在新仓库根目录运行。首条使用训练区第 1500 号指令检查真实搜索、详情页、购买得分与重置重放。服务只绑定本机，训练时需保持运行。`prepare_webshop.py` 从服务冻结任务 ID、目标顺序的 SHA256 和数据文件 SHA256。不同目录或顺序会让启动检查失败。

```bash
python scripts/check_webshop_live.py --query bag
python scripts/prepare_webshop.py --out artifacts/webshop
python run.py --size 4b --env webshop --method gigpo --model-dir metadata/Qwen3.5-4B --check
export MODEL_DIR=/mnt/models/Qwen3.5-4B
export CUDA_VISIBLE_DEVICES=0,1
python run.py --size 4b --env webshop --method grpo4 --smoke
python run.py --size 4b --env webshop --method ucf --smoke
```

本仓库固定 test 为顺序 ID 0–499，validation 为 500–599，train 为 1500 至最后一个 ID。验证保留区 600–1499 不加入训练；小型商品目录用于接口调试，正式对照使用上面的完整设置。测试只在完整训练结束时读取。组内训练奖励为「完整完成任务」的 0/1；成功率按官方购买得分是否等于 1 计算，`webshop_mean_score` 另报官方 0–1 部分得分。

## UCF 在哪里分叉

同网格一样，候选点在每回合第一个开围栏后。模型可以先推理，解析器执行第一个完整 `search[...]` 或 `click[...]` 指令块。WebShop 的可执行参数是开放文本，本版按首 token 的 **SEARCH / CLICK / RESIDUAL** 三类抽补集，后续参数由模型自然续写。类外概率质量一并计入。两次点击可以到完全不同的页面，当前分层没有把它们逐一枚举。GiGPO 用同题的可见页及可见操作摘要作为状态标识。模拟器按指令编号和已执行命令重放分叉，没有把商品答案、最优路径或奖励传给选点器。

随仓库的 CPU 测试用小型双商品 fixture 检查 HTTP、解析、局面重放、部分得分及 GiGPO。使用完整官方商品索引及实际 GPU 的通过条件分别是第一条现场验收和真实一轮更新。
