# 本地掌静脉研究

此目录是本地工作区脚本的发布副本。请按[仓库首页](../README.md#conditional-flow-and-dino--条件-flow-与-dino)的步骤，将其内容复制到名为 `LSAP-PV` 的仓库目录上一级，再执行脚本。下文目录和命令均以还原后的工作区根目录为基准，不应直接在 `local_workspace/` 内运行。下载缓存、权重、运行记录 JSON 和历史对话归档均不随源码发布。

当前目标：先复现原版 LSAP-PV 的一张图生成，再开展条件 Flow Matching、DINO 表征对齐和表征自条件的逐项实验。

## DINO 特征预测与反馈：已实现

已在 Flow 接口中加入冻结的官方 DINOv2 ViT-S/14 教师、局部特征预测和自条件反馈，同时保留纯 Flow 与仅特征对齐模式。默认纯 Flow 行为不变。

```powershell
.\run_dino_smoke.ps1
```

本机教师源码和权重已准备好；新环境首次运行可执行 `.\setup_dino.ps1`。试跑会训练两个更新、恢复后继续到第 3 步，并进行 20 步采样。**使用的图像仍为模拟数据，尚未进行真实掌静脉训练。** 采样阶段不加载 DINO 教师。

已验证官方教师冻结、特征及反馈梯度、无梯度预预测、零反馈分支、训练恢复与连续训练一致、无教师采样；完整检查共 14 项通过。默认 128×128 反馈模式在 RTX 3050 上训练的 PyTorch 分配显存峰值约 0.722 GiB；AMP＋激活检查点组合约 0.520 GiB，亦已验证训练、恢复与采样。

详细配置和真实数据命令见 [DINO_README.md](../DINO_README.md)。

## 纯条件 Flow：已实现

已新增独立的纯条件 Flow 单卡训练、恢复和 20 步欧拉采样入口，沿用静脉条件与风格标签；作为默认模式和 DINO 模式的对照。

在本目录运行：

```powershell
.\run_flow_smoke.ps1
```

此命令自动使用 `.venv`，用程序构造的模拟配对数据训练两个更新、恢复后再训练一个更新，最后生成一张图片。**这只是软件流程验证，模拟数据权重不能作为掌静脉生成研究结果。** 目前缺少真实训练数据，尚未进行真实数据的 Flow 训练。

默认 128×128、64 通道、batch size 为 1 已在 RTX 3050 上完成训练与采样。PyTorch 分配训练显存峰值约 0.636 GiB；另已验证 AMP、激活检查点和梯度累积组合。七项自动检查通过，包括恢复训练与连续训练参数一致，以及条件、风格实际影响输出。

真实数据格式、完整训练命令、模型定义、检查点和验证说明见 [FLOW_README.md](../FLOW_README.md)。环境安装与启动脚本在本目录随源码发布。

## 目录

- `LSAP-PV/`：从[你的仓库](https://github.com/wangqiyue70-droid/LSAP_PV-Layered-Spectral-Absorption-Projection)下载的 `codex/lsap-single-gpu-smoke` 分支，起始提交 `abb7db64e5fcc55a0a852820fc7e5abc250ec70e`。
- `.venv/`：本目录专用 Python 3.12 环境，不继承全局 site-packages。
- `LSAP-PV/checkpoint/`：作者原版预训练权重。
- `LSAP-PV/smoke_conditions/`：一个身份、一个样本的静脉树、LSAP 投影和预处理结果。
- `LSAP-PV/results_smoke/`：实际生成图像。
- `LSAP-PV/logs/`：采样日志。
- `.downloads/`：从官方 PyTorch 源下载并校验的 CUDA wheel 及断点文件。

## 原版 Diffusion 运行

在本目录的 PowerShell 中执行：

```powershell
.\run_local.ps1
```

脚本使用 `.venv\Scripts\python.exe`，无需手动激活环境。默认 GPU 为 `cuda:0`，`batch_size=1`，固定随机种子为 0，使用 **20 步重采样 DDPM** 验证代码链路，输出一张 128×128 灰度图。每次运行使用带时间的输出名，并写入同名 JSON 记录耗时、配置和文件哈希。

20 步模式保留原模型的 1000 步基础噪声调度，通过仓库已有的 `SpacedDiffusion` 选择覆盖完整过程的时间点、重新计算转移，再映射回模型原时间。它没有截断前 20 步，也没有将预训练模型的基础时间直接改为 20。快速结果仅用于功能检查，不能替代完整采样的质量评估。

指定种子、输出文件或使用 CPU：

```powershell
.\run_local.ps1 -Seed 1 -Output results_smoke/sample_seed1.png
.\run_local.ps1 -Device cpu -Output results_smoke/sample_cpu_new.png
.\run_local.ps1 -Steps 1000 -Output results_smoke/sample_full.png
```

现有条件图会被复用。如果希望重新制作条件，请直接在 `LSAP-PV` 目录运行：

```powershell
..\.venv\Scripts\python.exe generate_condition_single.py --output_dir smoke_conditions_new
```

新条件需要通过 `sample_single.py --cond_path smoke_conditions_new/pv_pattern_results/palmvein` 显式指定。生成器不会覆盖已有条件目录或输出图片。

## 环境准备与来源

环境已在本地配置过程中准备。需要重新安装时运行：

```powershell
.\setup_local.ps1
```

此脚本针对 Windows x64、CPython 3.12、NVIDIA GPU。使用 PyTorch 2.8.0 + CUDA 12.8、torchvision 0.23.0，版本组合参照 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/#v280)。wheel 的 SHA256 与官方软件索引比对。其他依赖见 `requirements-local.txt`；验证后的完整版本记录见 `research/requirements-lock.txt`。

模型权重来自原仓库 README 的 [DM-ckp](https://drive.google.com/file/d/1K85P8HTwGZ99v5jTL9dfOtZzlwZ6kBE8/view)。下载归档的 SHA256 记录为 `0c29a88f1a4651027db02f4437ec4df33d512a83b242916b1e3e700904f58a87`；这是本次下载的复现指纹，模型通过 `weights_only=True` 加载并严格匹配全部参数。

本地修改：`LSAP-PV/pvtree/process.py` 使用 `os.path.basename` 提取文件名，兼容 Windows 的反斜杠路径，并关闭图像文件、对短暂的目录占用做有限重试；`sample_single.py` 增加采样步数和运行记录。生成模型结构、权重、分辨率、条件输入和基础噪声调度沿用原分支，默认推理步数按本次需求减少为 20。

## 验证范围

硬件检测为 RTX 3050 Laptop GPU，显存 4GB。本机原有 PyTorch 是 CPU 版；研究环境单独安装 CUDA 版。

已实际完成：原 Python 环境的 CPU 完整 1000 步采样（约 5 分 4 秒，`LSAP-PV/results_smoke/sample_cpu.png`），以及项目 `.venv` 的 GPU 20 步快速采样（采样本身约 2.32 秒，PyTorch 分配显存峰值约 0.135 GiB，`LSAP-PV/results_smoke/sample.png`）。后者不包含 Python 启动和权重加载时间；显存数值不含全部 CUDA 运行时占用。这两次运行的设备和步数均不同，不能据此单独归因加速来源。

本节记录的是预训练原版的条件制作与推理验证。Flow 和 DINO 分支目前仅完成模拟数据软件测试，尚未进行真实数据训练，也未验证 FID、结构保真或下游识别性能；这些实验需要另外准备真实数据、身份划分及评估协议。原版详细运行结果以 `research/baseline_run.json` 和采样日志为准。
