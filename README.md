# Video Model LoRA Training Pipeline

A complete pipeline for training LoRA adapters on video generation models. Extract frames from videos, train style LoRA weights, and generate new video content.

## Project Structure

```
├── data/
│   ├── raw/              # Raw downloaded videos
│   ├── cropped/          # Extracted frames (for standard LoRA)
│   ├── temporal/         # Frame sequences (for temporal LoRA)
│   ├── framediff/        # Frame pairs (for frame-diff LoRA)
│   └── filtered/         # Manually curated training data
├── output/
│   ├── models/           # Trained LoRA weights
│   └── videos/           # Generated test videos
├── prompts/              # Prompt templates
│   ├── positive_prompts.txt
│   ├── negative_prompts.txt
│   └── video_prompts.txt
├── train.sh              # One-command training script
├── collect.py            # Video downloader (YouTube, Xvideos, SpankWire, Reddit)
├── preprocess.py         # Basic preprocessing (frame extraction, cropping)
├── preprocess_enhanced.py# Enhanced preprocessing (3 modes: images/temporal/framediff)
├── train_lora.py         # Standard image-based LoRA training (diffusers + PEFT)
├── train_video_lora.py   # Advanced temporal/frame-diff LoRA training
├── generate.py           # LoRA inference and test generation
├── train_config.yaml     # Training configuration
├── requirements.txt
└── README.md
```

## Quick Start

### Step 1: Setup Environment with uv

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use the bundled version
export PATH="/path/to/.uv:$PATH"

# Create virtual environment and install dependencies
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt

# Alternatively, let uv manage everything
uv sync
```

### Step 2: Collect Training Data

```bash
# Download videos from various sources
python collect.py --source xvideos --keywords "bikini,lingerie,wet swimsuit" --count 20 --output data/raw
python collect.py --source youtube --keywords "slow motion,beautiful woman" --count 20 --output data/raw
```

### Step 3: Preprocess

```bash
# Mode A: Standard image-based training (recommended for beginners)
python preprocess_enhanced.py --mode images --input data/raw --output data/cropped --fps 8

# Mode B: Temporal sequence training (advanced - learns motion patterns)
python preprocess_enhanced.py --mode temporal --input data/raw --output data/temporal --fps 8 --seq_length 8

# Mode C: Frame-diff training (advanced - learns frame transitions)
python preprocess_enhanced.py --mode framediff --input data/raw --output data/framediff
```

### Step 4: Train LoRA

```bash
# Standard LoRA (images)
bash train.sh

# Temporal LoRA (frame sequences)
python train_video_lora.py --mode temporal --video_dir data/temporal --output_dir output/models/temporal --epochs 20

# Frame-diff LoRA
python train_video_lora.py --mode frame_diff --video_dir data/framediff --output_dir output/models/framediff --epochs 20
```

### Step 5: Test Generation

```bash
# Generate test images/videos with trained LoRA
python generate.py --lora_path output/models/final_lora --output output/videos
```

## Training Modes

| Mode | Training Data | What It Learns | Difficulty | Best For |
|------|--------------|----------------|------------|----------|
| **Standard** | Static images | Appearance & style | Easy | Beginners, style transfer |
| **Temporal** | Frame sequences (8 frames) | Motion patterns & temporal relations | Advanced | Natural-looking motion |
| **Frame-Diff** | Adjacent frame pairs | Frame-to-frame transformations | Intermediate | Lightweight style adaptation |

## Hardware Requirements

- **GPU**: NVIDIA RTX 3060 or better (12GB VRAM recommended)
- **RAM**: 16GB+
- **Disk**: 20GB+ free space
- **Training Time**: 4-12 hours (depends on data size and GPU)

## Notes

1. **Data quality matters more than quantity** - 100 high-quality frames beat 500 mediocre ones
2. **Avoid overfitting** - If outputs look identical, reduce epochs or lower rank
3. **LoRA weight** - Start with 0.5 weight on the target platform, adjust gradually
4. **Style consistency** - Keep training data consistent (all real, or all anime, etc.)
