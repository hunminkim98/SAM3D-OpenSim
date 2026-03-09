# SAM3D-OpenSim Setup Guide

Complete installation instructions for the SAM3D Body to OpenSim pipeline.

## Prerequisites

- Windows 10/11
- Python 3.11
- CUDA-capable GPU (8GB+ VRAM recommended)
- Anaconda or Miniconda
- Git

## Step 1: SAM3D Body Environment

The pipeline requires the SAM3D Body environment with PyTorch and CUDA support.

```bash
# Create environment (if not exists)
conda create -n sam_3d_body python=3.11
conda activate sam_3d_body

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 2: Install SAM3D Body

```bash
# Clone SAM3D Body
cd C:\Sam3dBody
git clone https://github.com/facebookresearch/sam-3d-body.git

# Install
cd sam-3d-body
pip install -e .

# Download model weights (follow SAM3D Body instructions)
```

## Step 3: Install SAM3 (Segment Anything 3)

SAM3 is required for the `--detector sam3` option (recommended for best quality).

```bash
conda activate sam_3d_body

# Clone SAM3
cd C:\Sam3dBody
git clone https://github.com/facebookresearch/sam3.git

# Install SAM3
cd sam3
pip install -e .

# Install SAM3 dependencies for Windows
pip install decord psutil ftfy regex triton-windows
```

**Note:** SAM3 has a nested structure: `C:\Sam3dBody\sam3\sam3`

## Step 4: Install MoGe2 (FOV Estimation)

MoGe2 should be installed with SAM3D Body. Verify:

```bash
python -c "from moge.model.v2 import MoGeModel; print('MoGe2 OK')"
```

If not installed:
```bash
pip install moge
```

## Step 5: Verify All Imports

```bash
cd C:\Sam3DBodyToOpenSim
python test_imports.py
```

Expected output:
```
1. Testing SAM3 import...
   [OK] SAM3 imports successful
2. Testing MoGe2 import...
   [OK] MoGe2 imports successful
3. Testing FOVEstimator from SAM3D Body...
   [OK] FOVEstimator import successful
4. Testing HumanDetector from SAM3D Body...
   [OK] HumanDetector import successful
```

## Step 6: Pose2Sim Environment (OpenSim IK)

Create a separate environment for OpenSim inverse kinematics:

```bash
# Create Pose2Sim environment
conda create -n Pose2Sim python=3.11
conda activate Pose2Sim

# Install Pose2Sim with OpenSim
pip install pose2sim

# Verify OpenSim
python -c "import opensim; print(f'OpenSim version: {opensim.GetVersion()}')"
```

## Step 7: Install Blender (FBX Export)

1. Download Blender 5.0+ from https://www.blender.org/download/
2. Install to `C:\Program Files\Blender Foundation\Blender 5.0\`
3. Verify:
```bash
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --version
```

## Step 8: Clone This Repository

```bash
cd C:\
git clone https://github.com/AitorIriondo/SAM3D-OpenSim.git Sam3DBodyToOpenSim
cd Sam3DBodyToOpenSim
```

## Step 9: Install Python Dependencies

```bash
conda activate sam_3d_body
pip install -r requirements.txt
```

## Directory Structure

After setup, you should have:

```
C:\Sam3dBody\
├── sam-3d-body/           # SAM3D Body repository
├── sam3/                  # SAM3 repository
│   └── sam3/              # SAM3 package (nested)
└── checkpoints/           # Model weights
    └── sam-3d-body-dinov3/
        ├── model.ckpt
        └── assets/
            └── mhr_model.pt

C:\Sam3DBodyToOpenSim\     # This repository
├── config/
├── models/
├── src/
├── utils/
├── scripts/
└── ...
```

## Troubleshooting

### CUDA not available

```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with matching CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### SAM3 import errors

```bash
# Install missing dependencies
pip install decord psutil ftfy regex triton-windows
```

### OpenSim not found

```bash
# Ensure you're in the Pose2Sim environment
conda activate Pose2Sim
pip install pose2sim
```

### Blender not found

Update the path in `run_full_pipeline.py` and `run_export.py`:
```python
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
```

## Next Steps

After setup, see [HOW_TO_RUN.md](HOW_TO_RUN.md) for usage instructions.
