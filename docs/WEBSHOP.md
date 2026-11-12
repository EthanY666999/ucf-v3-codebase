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
python "$UCF_CODE/scripts/configure_webshop_full.py" "$PWD"
bash setup.sh -d all
PYTHONPATH="$PWD" python "$UCF_CODE/scripts/webshop_server.py" --port 8765
```

上游即使下载了全量文件，默认仍指向 1,000 件商品。配置脚本只在指定的官方 checkout 中切换商品与属性两个文件名，需在构建索引前执行。服务启动时会比对商品数和 Lucene 文档数，并将商品 ID 哈希写入目录身份。

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

每个回合开头都可以入选。原始轨迹和重掷都执行第一个完整的 `search[...]` 或 `click[...]` 指令块，再比较页面文字和已选规格。桥接响应的 `selected_options` 复制自上游 `env.server.user_sessions[str(goal_id)]["options"]`；商品选项点击更新这张“规格名 → 所选值”表，上游也用它给购买打分。它只参与后果比较，策略输入和 GiGPO 的可见状态标识维持原协议。训练端恢复回合起点时，按指令编号和已执行命令重放历史。[上游 SimServer 的规格记录和购买评分](https://github.com/princeton-nlp/WebShop/blob/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd/web_agent_site/envs/web_agent_text_env.py#L370-L405)

随仓库的 CPU 测试用小型双商品 fixture 检查 HTTP、解析、局面重放、部分得分及 GiGPO。使用完整官方商品索引及实际 GPU 的通过条件分别是第一条现场验收和真实一轮更新。
