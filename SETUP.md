# SAM3D-OpenSim Setup Guide

Current setup instructions for the canonical two-stage pipeline.

## Prerequisites

- Windows 10/11
- Python 3.11
- Anaconda or Miniconda
- Git
- CUDA-capable GPU recommended for Stage 1

## Step 1: Create the `sam_3d_body` environment

```bash
conda create -n sam_3d_body python=3.11
conda activate sam_3d_body

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

## Step 2: Install SAM3D Body

```bash
cd C:\Sam3dBody
git clone https://github.com/facebookresearch/sam-3d-body.git
cd sam-3d-body
pip install -e .
```

Download the model weights required by SAM3D Body and place them under your `checkpoints` directory.

## Step 3: Install SAM3

SAM3 is required for the `--detector sam3` path.

```bash
conda activate sam_3d_body
cd C:\Sam3dBody
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install decord psutil ftfy regex triton-windows
```

Note:

- the repo root is typically `C:\Sam3dBody\sam3`
- upstream still contains a nested `sam3` package inside that repo

## Step 4: Verify MoGe2

MoGe2 should already be available in the inference environment when SAM3D Body is set up correctly.

```bash
conda activate sam_3d_body
python -c "from moge.model.v2 import MoGeModel; print('MoGe2 OK')"
```

## Step 5: Clone this repository

```bash
cd C:\
git clone https://github.com/AitorIriondo/SAM3D-OpenSim.git Sam3DBodyToOpenSim
cd Sam3DBodyToOpenSim
```

## Step 6: Install repository-side dependencies

```bash
conda activate sam_3d_body
pip install -r requirements.txt
```

## Step 7: Verify imports and CLI surface

```bash
conda activate sam_3d_body
python test_imports.py
python run_inference.py --help
python run_export.py --help
python run_full_pipeline.py --help
```

## Step 8: Create the `Pose2Sim` environment

OpenSim inverse kinematics should run in a separate environment.

```bash
conda create -n Pose2Sim python=3.11
conda activate Pose2Sim
pip install pose2sim
python -c "import opensim; print(opensim.GetVersion())"
```

## Step 9: Install Blender

Blender is used for FBX export.

Recommended:

1. install Blender 4.5+ under `C:\Program Files\Blender Foundation\Blender *\`
2. verify it from a shell:

```bash
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --version
```

The runtime will auto-detect Blender from standard install locations.

If Blender lives somewhere else, set one of these environment variables instead of editing Python files:

```bash
set BLENDER_PATH=C:\Path\To\blender.exe
```

or

```bash
set SAM3D_OPENSIM_BLENDER_PATH=C:\Path\To\blender.exe
```

## Step 10: Expected external layout

```text
C:\Sam3dBody\
├── sam-3d-body\
├── sam3\
│   └── sam3\
└── checkpoints\
    └── sam-3d-body-dinov3\
        ├── model.ckpt
        └── assets\
            └── mhr_model.pt

C:\Sam3DBodyToOpenSim\
├── config\
├── docs\
├── models\
├── scripts\
├── src\
└── utils\
```

## Troubleshooting

### CUDA not available

```bash
nvidia-smi
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### SAM3 import errors

```bash
pip install decord psutil ftfy regex triton-windows
```

### OpenSim not found

```bash
conda activate Pose2Sim
pip install pose2sim
```

### Blender not found

Check the standard install path first. If Blender is installed elsewhere, set `BLENDER_PATH` or `SAM3D_OPENSIM_BLENDER_PATH`.

## Next Steps

- see `HOW_TO_RUN.md` for command examples
- prefer the two-stage workflow when iterating on export behavior
