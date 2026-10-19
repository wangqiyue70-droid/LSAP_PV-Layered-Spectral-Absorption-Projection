# 原版 LSAP-PV：先生成 1 张图

目标是跑通原版推理链路：**原版静脉树与 LSAP 投影 → 原版预处理 → 原版预训练条件扩散模型 → 1 张掌静脉图像**。不训练模型，不引入 Flow 或 DINO。

新增入口使用单卡、`batch_size=1`、128×128 灰度图、FP32 和原版完整 1000 步 DDPM。网络、权重格式、cosine 噪声调度和条件输入与原仓库采样配置保持一致。不会为节约显存改变网络宽度或分辨率。

## 1. 环境

以下命令在仓库根目录运行。建议使用独立 Python 环境。先根据机器的驱动安装匹配的 **PyTorch 和 torchvision**，参考 [PyTorch 官方安装入口](https://pytorch.org/get-started/locally/)。PyTorch 需支持 `torch.load(..., weights_only=True)`。

检查实际使用的解释器和 GPU：

```bash
python -c "import sys, torch, torchvision; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

安装这条最小生成链路需要的其他依赖：

```bash
python -m pip install numpy "Pillow>=9.1" tqdm matplotlib opencv-python bezier
```

如果已有条件图、只运行 `sample_single.py`，只需 PyTorch、numpy、Pillow 和 tqdm；该入口不依赖 MPI、blobfile、CycleGAN 或 torchvision。完整训练仍需原项目其他依赖。

## 2. 获取原版权重

下载原 README 的 [DM-ckp](https://drive.google.com/file/d/1K85P8HTwGZ99v5jTL9dfOtZzlwZ6kBE8/view?usp=share_link)，解压后找到 `ema_0.9999_070000.pt`。

原 README 写的是 `checkpoints`，原采样脚本写的是 `checkpoint`，两者并不一致。新入口要求显式提供实际 `.pt` 文件路径，不依赖目录名。下面假设你将文件放在：

```text
checkpoint/vein_sty_concat_sigma_cosine_vpem/ema_0.9999_070000.pt
```

权重不在 Git 仓库内，必须另行下载。仅测试预训练生成不需要下载真实掌静脉训练集或 CycleGAN 权重。

## 3. 生成 1 个条件图

```bash
python generate_condition_single.py
```

它按顺序调用原版 `get_bezier.py`、`main.py` 和 `process.py`，只生成 case 1 的 1 个身份、1 个样本。静脉树复杂度与原版默认值一致；仍执行 ROI 裁剪、掌纹线条混合及原版增强。所有中间文件放入新建的 `smoke_conditions` 目录。

条件图位于 `smoke_conditions/pv_pattern_results/palmvein` 的子目录中。子目录末尾的 `_0` / `_1` / `_2` / `_3` 是原预处理随机分配的数据集风格标签。

已有原版生成的条件图时，可跳过这一步。重新运行时用新的目录，例如 `python generate_condition_single.py --output_dir smoke_conditions_2`，然后相应调整下一条命令的条件路径。

## 4. 单卡生成 1 张掌静脉图

```bash
python sample_single.py --model_path checkpoint/vein_sty_concat_sigma_cosine_vpem/ema_0.9999_070000.pt --cond_path smoke_conditions/pv_pattern_results/palmvein --batch_size 1
```

入口会递归选择条件目录中排序后的第一张图片，自动读取其直接父目录的风格标签，显示 1000 步采样进度，保存 `results_smoke/sample.png`。输出是 128×128 灰度 PNG。采样结束会报告 PyTorch 的显存分配峰值；它不等于进程全部显存占用。

也可以直接指定一张条件图，并明确选择风格：

```bash
python sample_single.py --model_path /path/to/ema_0.9999_070000.pt --cond_path /path/to/condition.png --style_id 0 --batch_size 1 --output results_smoke/sample_casia.png
```

风格对应原训练代码：`0=casia`、`1=HFUT`、`2=polyu`、`3=TongJi`。输入应是原 LSAP/预处理得到的条件图，不是任意真实照片。自定义路径含空格时，请用双引号包住路径。

默认设备为 `cuda:0`；可用 `--device cuda:1` 选择另一张可见 GPU。CPU 仅用于排查，可加 `--device cpu`，完整 1000 步可能很慢。再次生成请指定不同的 `--output`，脚本不会覆盖已有图片。

## 怎样判断跑通

1. 条件生成命令输出 `Condition ready`，条件图能正常打开。
2. 模型权重严格加载成功，无 missing/unexpected keys 或尺寸不匹配。
3. 1000 步完成，输出 `Saved`，`results_smoke/sample.png` 能正常打开。

这验证的是原版预训练模型的生成链路，不代表训练链路、FID 或识别性能已经复现。`batch_size=1` 降低单次采样的激活显存需求，但不降低模型权重占用，也不缩短 1000 步的长度；所需显存和时间需在实际机器上测量。

若报错，请保留完整 traceback，并附上第一步的环境检查输出。缺文件、CUDA 不可用、输出路径已存在会在采样前提示；checkpoint 尺寸不匹配时应先核实权重来源，不要修改网络配置来强行加载。
