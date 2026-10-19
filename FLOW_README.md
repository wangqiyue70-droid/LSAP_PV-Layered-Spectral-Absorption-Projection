# 纯条件 Flow Matching

本模块用条件 Flow Matching 替换 LSAP-PV 的扩散生成阶段。保留原来的静脉条件制作和四种数据集风格，复用原项目 U-Net 主体；新增独立的单卡训练和欧拉采样入口，无需 MPI、blobfile 或新的 Python 依赖。

**当前已验证的是软件链路。现有 Flow 试跑权重只在程序生成的模拟数据上更新了几步，不能用于评价掌静脉生成质量。尚未进行真实数据训练。** 默认仍为纯 Flow；新增 DINO 对齐与反馈模式见 [DINO_README.md](DINO_README.md)。

## 1. 最快验证

在上一级本地研究目录运行：

```powershell
cd 'D:\文档\ChatGPT\palm'
.\run_flow_smoke.ps1
```

脚本自动使用项目 `.venv`，依次执行：

1. 用模拟数据训练两个优化器更新，保存第 2 步检查点。
2. 加载模型、优化器、数据游标和随机状态，再训练一个更新，保存第 3 步。
3. 使用已有 LSAP 条件图进行 20 步欧拉采样，保存 PNG 和同名 JSON。

输出位于 `LSAP-PV/runs/flow_smoke_时间/` 和 `LSAP-PV/results_flow/`。每次默认使用新的时间命名，避免覆盖。CPU 排查可运行 `..\run_flow_smoke.ps1 -Device cpu`（在本仓库目录内），也可在上一级运行 `.\run_flow_smoke.ps1 -Device cpu`。

模拟数据由程序画出弯曲线条，并构造带灰度背景和轻微噪声的配对目标图。它仅用于检验损失、反向传播、参数更新、保存恢复和采样；不是实际采集的掌静脉，也不是原 Diffusion 模型生成的数据。

只有显式指定 `--smoke-test` 才能新建模拟数据训练；没有数据时不会自动偷偷切换。检查点、日志、采样 JSON 都保存 `synthetic_smoke` 标记，恢复时不能混用真实数据与模拟数据。

## 2. Flow 的定义

时间从噪声走向真实图像：

- `t=0`：标准高斯噪声 ε。
- `t=1`：真实图像 x。
- 中间状态：`x_t = (1-t) ε + t x`。
- 目标速度：`v_target = x - ε`。
- 损失：预测速度与目标速度之间的像素均方误差。

网络输入为中间图像、连续时间、静脉条件和风格标签；输出为单通道速度。默认 128×128、基础通道数 64、两个残差块、四个注意力头，沿用原 128×128 U-Net 的通道倍率。图像与条件按通道拼接；风格嵌入明确设为 4 类：`0=casia`、`1=HFUT`、`2=polyu`、`3=TongJi`。

连续时间乘以 1000 后送入正弦时间嵌入。这只是嵌入尺度，不是扩散时间调度，也不要求采样 1000 步。采样用欧拉法从 0 积分到 1，每步一次网络调用，默认 20 步；中间状态不裁剪，仅在输出 PNG 时转换和裁剪到像素范围。

本实现采用噪声与图像独立配对的直线路径，不进行额外的最优传输配对。方法背景见 [Flow Matching 原论文](https://arxiv.org/abs/2210.02747)。

## 3. 真实训练数据

当前本地尚无真实训练数据。准备好真实掌静脉 ROI 及其对应 VPEM 条件后，按以下结构存放：

```text
traindata/train/
  vein/
    casia_subject001/
      image001.png
      image002.png
  cond/
    casia_subject001/
      image001.png
      image002.png
```

`vein` 与 `cond` 按相对路径和不含扩展名的文件名配对，允许 `.jpg` 对 `.png`，禁止缺配对或重复样本键。图像与条件的原始空间尺寸必须一致；它们应对应同一 ROI，单纯同尺寸不代表已对齐。

默认根据真实图像直接父目录的 `casia_`、`HFUT_`、`polyu_`、`TongJi_` 前缀识别风格，大小写不敏感。单一风格且目录不采用这些名称时，可显式使用 `--style-id 0` 等。

图像和条件转换为灰度并归一化到 `[-1,1]`。默认使用配对同步的缩放和随机裁剪，轻微亮度、对比度扰动仅用于真实图像；可使用 `--no-augment` 关闭增强。

请在训练前按身份划分训练、验证、测试集，`--data-dir` 只指向训练部分。当前入口不会自动划分身份，也不执行验证集选择或识别指标评估。

## 4. 训练与恢复

以下命令均在 `LSAP-PV` 仓库目录执行，使用上一级已安装的虚拟环境：

```powershell
..\.venv\Scripts\python.exe train_flow.py `
    --data-dir traindata/train `
    --out-dir runs/flow_real `
    --steps 1000 --batch-size 1 --device cuda:0
```

`--steps` 是优化器更新总数，不是 epoch 数或采样步数。`--grad-accum 8` 表示每次更新累积八个微批次；默认学习率为 `2e-4`，使用 AdamW 和梯度范数裁剪。第一版未加入 EMA。

每 `--save-every` 个更新及最后一个更新保存检查点，默认间隔为 1000。新训练要求输出目录为空，不会覆盖已有运行。

例如从第 1000 步继续到第 2000 步：

```powershell
..\.venv\Scripts\python.exe train_flow.py `
    --resume runs/flow_real/flow_step_0001000.pt `
    --out-dir runs/flow_real --steps 2000 --device cuda:0
```

恢复时使用保存的模型和训练设置。数据目录移动后可重新提供 `--data-dir`；程序会核对相对路径、文件大小和修改时间的指纹。该指纹不等于逐文件内容哈希。修改批量大小、增强、学习率或网络配置需要建立新运行，不会在恢复时悄悄改变。

检查点包含模型、优化器、AMP scaler、训练步数、数据类型和指纹、打乱顺序及游标、Torch CPU / 当前 GPU 随机状态。已验证 CPU 上同配置恢复与连续训练得到逐参数相同的结果；跨设备或不同 CUDA 实现不承诺逐位一致。

### 4GB 显卡选项

默认 128×128、通道数 64、batch size 为 1 已在本机 RTX 3050 4GB 上完成反向传播。需要降低显存时，可在新运行中加上：

```powershell
--amp --activation-checkpoint --grad-accum 8
```

AMP 使用 CUDA FP16 与 GradScaler；激活检查点采用 PyTorch 的非重入接口，保留重计算的 autocast 和 dropout 随机状态。训练入口会统计真实显存峰值。`--model-channels 32` 或 `--image-size 32/64` 可用于更小的工程试跑，但这些配置与默认模型规模不同，正式对照应匹配设置。

## 5. 从 Flow 权重采样

```powershell
..\.venv\Scripts\python.exe sample_flow.py `
    --model-path runs/flow_real/flow_step_0001000.pt `
    --cond-path smoke_conditions/pv_pattern_results/palmvein `
    --steps 20 --seed 0 --device cuda:0 `
    --output results_flow/sample_real.png
```

条件路径可以是一张图或目录；目录按路径排序后取第一张。风格可从真实数据目录前缀或合成条件目录的 `_0` 至 `_3` 后缀读取，也可通过 `--style-id` 指定。

必须使用 `train_flow.py` 产生的 Flow 检查点。**作者原版 Diffusion 权重不能直接用作 Flow 权重**；加载时会校验独立格式并严格匹配模型参数。当前 Flow 从随机初始化开始训练。

输出的同名 JSON 包含训练数据类型、训练步数、模型配置、条件及权重哈希、种子、风格、网络调用次数、采样时间和显存。由模拟数据权重生成的图片会明确标记 `synthetic_smoke_only: true`。

## 6. 已完成的本机验证

固定运行目录 `runs/flow_smoke_verified/` 保存了默认 128×128、64 通道、FP32、batch size 为 1 的试跑：

- 两个优化器更新完成，并保存第 2 步检查点。
- 从第 2 步恢复，完成第 3 步并保存。
- 20 步欧拉采样出图，文件为 `results_flow/flow_smoke_verified.png` 及同名 JSON。
- 默认训练的 PyTorch 分配显存峰值约 0.636 GiB。

`runs/flow_smoke_amp/` 另外验证了 AMP、激活检查点和两次梯度累积的组合，PyTorch 分配显存峰值约 0.431 GiB。以上均为模拟数据软件测试；显存不包含进程全部 CUDA 开销，耗时与显存会随环境和配置变化。

运行自动检查：

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

七项原 Flow 检查覆盖时间端点和目标速度、欧拉积分方向及网络调用次数、真实文件配对与归一化、缺配对和重复键拒绝、风格推断、恢复训练与连续训练的一致性、条件与风格实际影响输出，以及拒绝 Diffusion 权重。当前命令还会执行新增七项 DINO 检查，详见 DINO 说明。测试夹具只用于软件验证。

## 7. 新增文件

- `conditional_flow/model.py`：条件速度 U-Net 与可选激活检查点。
- `conditional_flow/matching.py`：直线路径和欧拉积分。
- `conditional_flow/data.py`：真实配对图像加载、显式模拟数据、可恢复的数据游标。
- `conditional_flow/runtime.py`：设备选择、独立检查点格式和运行记录。
- `train_flow.py`：单卡训练、梯度累积、AMP、保存和恢复。
- `sample_flow.py`：20 步默认采样、输出 PNG 和 JSON。
- `tests/test_conditional_flow.py`：数值与恢复检查。
- 上一级 `run_flow_smoke.ps1`：一键验证训练、恢复和采样。

原版 Diffusion 入口仍可使用。纯 Flow、DINO 对齐与特征反馈三种模式已经接入，下一阶段需要真实数据训练和对照评估。
