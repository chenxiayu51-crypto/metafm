# Windows 本地运行与服务器迁移

正式仓库 `C:/Users/18165/Documents/ChatGPT/metafmodel`。
已安装 editable metafmodel0.2.0，复用相邻MetaEMGFM/.venv，未复制CUDA环境。
可设置 META_PYTHON 或创建主工程 .venv；优先使用项目内环境，其次相邻环境。

```powershell
cd C:\Users\18165\Documents\ChatGPT\metafmodel
.\scripts\run_local.ps1 verify
.\scripts\run_local.ps1 test
.\scripts\run_local.ps1 pretrain data=local trainer=local_gpu +pretrain=local_wrist trainer.max_steps=10
```

新机器：创建Python环境，安装对应CUDA PyTorch后 `python -m pip install -e .[test]`。
Hydra训练会切换输出目录，CLI传入的data路径与checkpoint必须绝对路径。
local默认引用相邻旧工程；可通过META_CANONICAL_ROOT、META_MANIFEST_PATH、
META_CORPUS_PATH覆盖。统一验收用configs/local/local_verify.yaml，路径相对
正式仓库解析。local_verify每次唯一run目录和latest.json，失败阶段保留日志
并继续独立检查，只有FAIL返回非零；BLOCKED不是伪造PASS。

常见问题：GBK控制台用无进度条日志；CPU测试提示GPU未使用属测试配置；
EMA teacher eval warning属预期；Wrist无val不能保存best。仅加载可信本工程
checkpoint，训练入口允许完整Lightning状态反序列化。

A800：`data=a800 trainer=a100_ddp`是准备配置，尚未真实运行。
必须重核验GPU型号（用户称华为A800，现无法确认是否CUDA兼容）、驱动、
PyTorch/CUDA或实际NPU运行栈、bf16/DDP、workers、数据吞吐及显存。
单机多卡需关闭Lightning自动sampler：配置已use_distributed_sampler=false。
多Nature任务使用task→corpus映射。调大batch/窗口前重新测量；本地只验证
200ms与2s小模型，不能外推大模型8GB可运行。没有承诺服务器直接PASS。
