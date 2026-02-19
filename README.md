# Chess Vision Project

## Overview

This project implements a computer vision pipeline for chess board recognition and piece classification from photographs. The system consists of three main stages: (1) corner detection to identify the chessboard boundaries, (2) homography transformation and rectification to obtain a top-down view of the board, and (3) deep learning-based classification to identify each chess piece. The pipeline can process an input image and output the complete board state in standard chess notation, enabling applications such as automated game analysis, move validation, and digital board reconstruction.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/TalaalMaxood05/ChessVisionProject.git
cd ChessVisionProject
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages
The following packages will be installed:
- opencv-python==4.9.0.80 (Computer vision operations)
- torch==2.2.0 (Deep learning framework)
- torchvision==0.17.0 (Vision utilities for PyTorch)
- numpy==1.26.4 (Numerical computing)
- python-chess==1.999 (Chess logic and notation)
- stockfish==3.28.0 (Chess engine integration)
- matplotlib==3.8.0 (Visualization)

## Dataset Setup

### Downloading Datasets

This project uses two main datasets for training and evaluation:

1. **ChessReD Dataset**
   - Download from: https://github.com/bmazin/ChessReD
   - Extract to: `data/ChessReD/`

2. **Chesscog Dataset**
   - Download from: https://github.com/georgw777/chesscog
   - Extract to: `data/chesscog/`

### Important Notes
- Do not commit actual dataset files to the repository
- The `.gitignore` file is configured to exclude large dataset files
- See `data/README.md` for detailed dataset instructions

## Usage

### Running the Complete Pipeline

To process a chess board image through the complete pipeline:

```bash
python src/pipeline.py --image photo.jpg
```

### Command-Line Options

- `--image`: Path to the input chess board image (required)
- `--model`: Path to the trained classification model (default: `models/best_model.pth`)
- `--output`: Path to save output results (optional)

### Example

```bash
python src/pipeline.py --image examples/game1.jpg --model models/best_model.pth --output results/game1_output.json
```

## Project Structure

```
chess-vision/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── data/                 # Datasets (not committed)
│   └── README.md         # Dataset download instructions
├── src/                  # Source code
│   ├── stage1_corners.py # Corner detection (Members A & B)
│   ├── stage2_warp.py    # Homography + rectification (Members A & B)
│   ├── stage3_classify.py # Piece classifier (Member C)
│   ├── pipeline.py       # Integration pipeline (Member A)
│   └── utils.py          # Shared helper functions
├── models/               # Saved trained models
├── experiments/          # Sensitivity analysis and comparisons
├── results/              # Output figures, tables, logs
└── report/               # LaTeX files for reports
```

## Development Workflow

### Branching Strategy

The project uses a simple branching model:

- `main` - Stable, working code only
- `member-a` - Stages 1-2 and pipeline integration
- `member-b` - Sensitivity experiments
- `member-c` - Piece classifier development

### Contributing

1. Create a branch for your work
2. Make changes and commit regularly
3. Open a Pull Request into `main`
4. At least one other team member reviews the PR
5. Merge after approval

This workflow prevents breaking changes and maintains code quality.

## Pipeline Stages

### Stage 1: Corner Detection
- **Responsibility**: Members A & B
- **File**: `src/stage1_corners.py`
- **Function**: Detects the four corners of the chessboard in the input image

### Stage 2: Homography & Rectification
- **Responsibility**: Members A & B
- **File**: `src/stage2_warp.py`
- **Function**: Applies homography transformation to obtain a top-down view of the board

### Stage 3: Piece Classification
- **Responsibility**: Member C
- **File**: `src/stage3_classify.py`
- **Function**: Uses a trained neural network to classify each chess piece

### Integration Pipeline
- **Responsibility**: Member A
- **File**: `src/pipeline.py`
- **Function**: Connects all three stages into a complete end-to-end system

## License

This project is for educational purposes as part of a computer vision course.

## Team Members

- Member A: Corner detection, homography, and pipeline integration
- Member B: Corner detection, sensitivity analysis
- Member C: Piece classification using deep learning