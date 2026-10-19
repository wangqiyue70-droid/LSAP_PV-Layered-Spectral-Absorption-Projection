# LSAP_PV-Layered-Spectral-Absorption-Projection
*LSAP-PV: High-Fidelity Palm Vein Image Synthesis via Layered Spectral Absorption Projection-Guided Diffusion Model*
| [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/37837)

## Single-GPU smoke run / 单卡最小试跑

See [RUN_SINGLE_GPU.md](RUN_SINGLE_GPU.md) for the original pretrained pipeline with one condition image, `batch_size=1`, and one generated image.

## Conditional Flow and DINO / 条件 Flow 与 DINO

新增纯条件 Flow Matching 训练、恢复和少步欧拉采样，并支持冻结的官方 DINOv2 教师、局部特征预测与自条件反馈。运行方式见 [FLOW_README.md](FLOW_README.md) 和 [DINO_README.md](DINO_README.md)。当前只完成模拟数据的软件验证，尚未进行真实掌静脉数据训练。

Windows 环境安装、下载、启动脚本和本地研究说明统一保存在 [local_workspace/](local_workspace/README.md)。这些脚本按工作区布局使用：仓库目录名为 `LSAP-PV`，`.venv`、`.downloads` 和启动脚本位于仓库上一级。新电脑可在 PowerShell 中执行：

```powershell
New-Item -ItemType Directory -Path palm-workspace
Set-Location palm-workspace
git clone --branch codex/lsap-single-gpu-smoke https://github.com/wangqiyue70-droid/LSAP_PV-Layered-Spectral-Absorption-Projection.git LSAP-PV
Copy-Item -Path .\LSAP-PV\local_workspace\* -Destination . -Recurse -Force
.\setup_local.ps1
.\run_flow_smoke.ps1
.\setup_dino.ps1
.\run_dino_smoke.ps1
```

恢复脚本只需在新工作区复制一次；已有工作区应先核对同名文件，再选择是否覆盖。`setup_local.ps1` 面向 Windows x64、Python 3.12 和 NVIDIA CUDA 环境，包含原版 Diffusion 权重下载；仅使用 Flow 的其他环境可按 Flow 文档自行配置。权重、虚拟环境、下载缓存及生成结果不随源码提交。

## Prerequisites
- Python 3
- NVIDIA GPU + CUDA CuDNN

# Getting Started

## Installation
- Clone this repo:
```bash
git clone https://github.com/Sunniva-Shang/LSAP_PV-Layered-Spectral-Absorption-Projection
``` 
- Install dependencies:
`
pip install -r requirements.txt
`

## Palm Vein Images Generation

### Palm Vascular Tree Generation
- `cd pvtree;
  bash get_patterns.sh`
- The output are saved in `./pvtree/pv_pattern_results/palmvein`

### Images Generation
- Download [DM-ckp](https://drive.google.com/file/d/1K85P8HTwGZ99v5jTL9dfOtZzlwZ6kBE8/view?usp=share_link), unzip it and place it in `./checkpoints`.
- `bash sample_muti_gpus.sh`
- The generated images are saved in `./results`.

## Train the Diffusion Model
### Preparing Training Data
- Download [CycleGAN-ckp](https://drive.google.com/file/d/1O9HVMWcOjLyVW26cvkO0OYdisJYYZW7d/view?usp=share_link), unzip it and place it in `./cyclegan/checkpoints`.
- Extracting palm vein patterns from real data with a pre-trained model `bash test.sh`
- The generated images are saved in `./cyclegan/results`.

### Storage Format of Training Data
<pre>
traindata             
├── cond                 
│   ├── {DatasetName}_{idname}  # DatasetName={casia/HFUT/polyu/TongJi}         
│   │   ├── img1.png/jpg/...   
│   │   └── ...   
│   └── ...              
└── vein                
    ├── {DatasetName}_{idname}             
    │   ├── img1.png/jpg/...
    │   └── ...          
    └── ... 
</pre>              

### Start Training Diffusion Model
- `bash train.sh`
- More details see paper [IDDPM](https://ojs.aaai.org/index.php/AAAI/article/view/28039).


### Citation
If you find this useful for your research, please use the following.

```
@inproceedings{shang2025pvtree,
  title={PVTree: Realistic and controllable palm vein generation for recognition tasks},
  author={Shang, Sheng and Zhao, Chenglong and Zhang, Ruixin and Jin, Jianlong and Zhang, Jingyun and Guo, Rizen and Ding, Shouhong and Wu, Yunsheng and Zhao, Yang and Jia, Wei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={6767--6775},
  year={2025}
}

@inproceedings{nichol2021improved,
  title={Improved denoising diffusion probabilistic models},
  author={Nichol, Alexander Quinn and Dhariwal, Prafulla},
  booktitle={International conference on machine learning},
  pages={8162--8171},
  year={2021},
  organization={PMLR}
}

```

If you have any questions or encounter any issues with the code, please feel free to contact me (email: shengshang@mail.hfut.edu.cn). 
I would be more than happy to assist you in any way I can.

### Acknowledgements
This code borrows heavily from the [PVTree](https://github.com/Sunniva-Shang/PVTree-palmvein-generation) and
[IDDPM](https://github.com/openai/improved-diffusion).
