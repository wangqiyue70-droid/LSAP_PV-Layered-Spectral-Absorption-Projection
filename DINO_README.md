# 条件 Flow＋DINO 特征预测与反馈

本模块在现有条件 Flow 上增加冻结的 DINOv2 教师、局部特征预测头及预测表征自条件。静脉条件、风格标签、图像速度目标和原来的纯 Flow 入口继续使用。

**已使用官方预训练 DINOv2 权重完成本机软件验证；训练图像仍是显式模拟数据。尚无真实掌静脉训练数据，因此这些试跑权重和图片不能用于评价生成质量、细分支保真或识别性能。**

## 1. 快速运行

在本地研究根目录运行：

```powershell
cd 'D:\文档\ChatGPT\palm'
.\run_dino_smoke.ps1
```

本机的教师源码和权重已准备好。首次在其他环境配置时，先运行：

```powershell
.\setup_dino.ps1
# 如需指定本机代理：
.\setup_dino.ps1 -Proxy http://127.0.0.1:7890
```

试跑使用项目 `.venv`，先训练两个更新，恢复后继续到第 3 步，再进行 20 步欧拉采样。结果位于 `LSAP-PV/runs/dino_smoke_时间/` 和 `LSAP-PV/results_flow/`。无需手动激活虚拟环境，也不会覆盖已有运行。

```powershell
# 带特征预测，但关闭反馈，用于消融对照
.\run_dino_smoke.ps1 -DinoMode align

# 带反馈，并开启混合精度和激活检查点
.\run_dino_smoke.ps1 -Amp -ActivationCheckpoint
```

## 2. 三种模式

- `--dino-mode none`：原纯条件 Flow，默认值；不加载 DINO 教师。
- `--dino-mode align`：Flow＋局部特征预测与对齐，不反馈预测特征。
- `--dino-mode feedback`：Flow＋局部特征预测、对齐及自条件反馈。

纯 Flow 的已有检查点仍可加载。模式是模型结构的一部分，恢复训练时不能切换；新建 DINO 实验需要新的输出目录。当前 DINO 模式从随机初始化训练生成器，不会自动将已有纯 Flow 或 Diffusion 权重转换为新模型。

## 3. 教师与目标特征

教师采用 [Meta 官方 DINOv2](https://github.com/facebookresearch/dinov2) 的 ViT-S/14，不含 register tokens，参数冻结并始终处于 eval 模式。源码固定在提交：

```text
7764ea0f912e53c92e82eb78a2a1631e92725fc8
```

源码位于上一级 `.downloads/dinov2/`，权重位于 `.downloads/dinov2_vits14_pretrain.pth`，来源为官方 `dl.fbaipublicfiles.com`。通过 `weights_only=True` 严格加载参数；源码及权重哈希记录在 `research/dino_teacher.json` 和训练检查点。权重哈希是本次下载的复现指纹，不是发布者提供的校验值。

教师输入来自**当前批次增强后的干净目标图像 x**，与该批次条件图对应。将灰度 `[-1,1]` 图像转换到 `[0,1]`、复制成 RGB、对完整 ROI 做带抗锯齿的 bicubic 缩放到 224×224，再按 ImageNet 均值和标准差归一化。这里不额外做中心裁剪，以保持整个 ROI 的空间对应。

使用最后一层的 `x_norm_patchtokens`，不使用 CLS token。目标形状为 `B×384×16×16`，即 16×16 个 patch、每个 patch 384 维。

第一版在线提取目标特征，保证随机裁剪和图像增强后的特征正确对应；没有启用离线缓存。可通过 `--dino-device cpu` 单独把教师放在 CPU，默认与生成器同设备。教师不进入生成器优化器，目标特征也不参与梯度传播。

## 4. U-Net 中的预测与反馈

默认图像为 128×128。在编码器的 16×16 尺度，将预测表征按通道归一化，再通过 1×1 投影加到 U-Net 特征中。投影不带 bias，零表征等效于零反馈。特征预测头从解码器相应尺度输出 384 通道局部特征；较小工程测试尺寸会插值到教师的 16×16 网格。

这是一种针对现有 U-Net 的适配。方法借鉴 [DINO Forcing](https://arxiv.org/abs/2610.11751) 的预测表征与反馈机制，没有复刻其 Transformer 架构、完整训练配方或 ImageNet 结果。

训练仍使用原 Flow 约定：`x_t=(1-t)ε+t x`，速度目标为 `x-ε`。联合损失为：

```text
L = MSE(v_pred, x-ε) + λ × mean_patch(1-cos(h_pred, h_teacher))
```

`--dino-weight` 控制 λ，默认 0.5。余弦损失沿每个 patch 的 384 个通道计算，再对批次和空间求均值。

反馈模式下：

1. 默认每个样本以 90% 概率使用反馈。对需要反馈的批次先做一次无梯度、零自条件的表征预预测。
2. 将预预测 detach，作为同一 `x_t`、时间、静脉条件及风格的第二次前向输入。
3. 有梯度的前向联合优化速度与特征损失。
4. 剩余样本使用零自条件完成有梯度训练，覆盖采样第一步的情况。

概率通过 `--self-cond-prob` 配置，范围 `[0,1]`。真实教师特征只作为监督目标，不直接作为自条件输入；`align` 模式没有额外预预测。

## 5. 真实数据训练

先按 [FLOW_README.md](FLOW_README.md) 准备真实图像与 VPEM 条件，并按身份划分数据。以下命令在 `LSAP-PV` 仓库内执行：

```powershell
..\.venv\Scripts\python.exe train_flow.py `
    --data-dir traindata/train `
    --dino-mode feedback --dino-weight 0.5 --self-cond-prob 0.9 `
    --batch-size 1 --steps 1000 --device cuda:0 `
    --out-dir runs/flow_dino_real
```

显存优化可在新运行中加入 `--amp --activation-checkpoint --grad-accum 8`。恢复时自动读取原模型模式、损失权重、反馈概率和优化器状态：

```powershell
..\.venv\Scripts\python.exe train_flow.py `
    --resume runs/flow_dino_real/flow_step_0001000.pt `
    --steps 2000 --out-dir runs/flow_dino_real --device cuda:0
```

恢复训练需要教师文件。程序比对教师权重、源码、特征定义和预处理的签名，拒绝悄悄更换监督目标。教师文件移动后可显式指定 `--dino-source` 和 `--dino-weights`，内容需与记录一致。CPU 上已验证恢复训练与连续训练逐参数一致；跨设备和不同 CUDA 实现不承诺逐位一致。

## 6. 采样不需要教师

```powershell
..\.venv\Scripts\python.exe sample_flow.py `
    --model-path runs/flow_dino_real/flow_step_0001000.pt `
    --cond-path smoke_conditions/pv_pattern_results/palmvein `
    --steps 20 --device cuda:0 --output results_flow/dino_sample.png
```

反馈采样从高斯噪声和零自条件开始。每步一次网络调用，同时得到速度与预测表征；用速度更新图像，将表征直接传到下一步。**20 步就是 20 次网络调用，只有图像一条 ODE，不对 DINO 特征做额外去噪或积分。**

采样入口不导入或构造教师；只需要生成器检查点与静脉条件。已用不存在的教师路径验证这一点。JSON 会记录模式、是否反馈、网络调用次数、教师来源签名以及 `teacher_required_at_inference: false`。教师权重不打包到生成器检查点。

程序接口继续支持 `model(x,t,condition,style)` 返回速度。特征模式下调用 `model.predict(..., self_condition=...)` 可同时取得 `.velocity` 与 `.representation`；自条件在接口内也会 detach。

## 7. 本机验证

在 `.venv`、RTX 3050 Laptop GPU、128×128、64 通道、batch size 为 1 上验证：

- 官方 DINOv2 参数严格加载，教师输出形状正确，参数冻结且无梯度。
- 默认反馈模式完成两步训练、从第 2 步恢复到第 3 步，以及 20 步采样。
- 执行了反馈与零反馈两个训练分支，分别记录速度损失、对齐损失和反馈比例。
- 固定试跑文件位于 `runs/dino_feedback_verified/` 和 `results_flow/dino_feedback_verified.png`。
- 此次默认训练的 PyTorch 分配显存峰值约 0.722 GiB，采样循环约 1.92 秒；不含启动、教师加载和生成器加载时间，也不代表完整 CUDA 显存占用。
- `runs/dino_align_verified/` 已验证仅特征对齐模式的训练、恢复及采样。
- `runs/dino_feedback_amp_fixed/` 已验证 AMP＋激活检查点的完整流程，训练分配显存峰值约 0.520 GiB。无梯度预预测关闭 autocast 参数缓存，避免将无梯度的半精度参数缓存复用到有梯度前向；回归检查确认主干参数仍有有效梯度。

运行检查：

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

已有 7 项纯 Flow 检查及新增 7 项 DINO 检查，共 14 项通过。覆盖特征损失、detach、预测头与反馈投影的梯度、预预测无梯度、零反馈训练、特征逐步复用、冻结教师、恢复一致性、无教师采样、监督目标签名，以及 AMP 预预测在有/无激活检查点时的主干梯度。未准备教师文件的环境会跳过两项官方教师集成检查；没有 CUDA 时另跳过 FP16 回归检查。

缺少 xFormers 的提示是官方 DINOv2 的可选加速提示，本机已经使用标准 PyTorch 路径验证，无需为此更换现有 PyTorch。

## 8. 文件变更

- `conditional_flow/model.py`：新增三种模式、局部预测头、反馈投影及双输出接口。
- `conditional_flow/matching.py`：增加局部余弦损失、无梯度预预测和采样表征逐步反馈。
- `conditional_flow/teacher.py`：离线加载并冻结官方 DINOv2，记录教师签名。
- `train_flow.py`：加入模式与损失参数、联合训练、教师一致性检查和分项日志。
- `sample_flow.py`：自动采用保存的模式，记录采样不依赖教师。
- `tests/test_dino_feedback.py`：新增机制与真实教师集成检查。
- 上一级 `setup_dino.ps1`、`research/setup_dino_teacher.py`：准备固定版本的教师。
- 上一级 `run_dino_smoke.ps1`：一键软件试跑。
