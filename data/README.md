# Dataset Instructions

## Where to Download Datasets

This project uses the following datasets:

### 1. ChessReD Dataset
- **Download from:** [ChessReD GitHub Repository](https://github.com/bmazin/ChessReD)
- **Description:** A dataset for chess piece recognition
- **Extract to:** `data/ChessReD/`

### 2. Chesscog Dataset
- **Download from:** [Chesscog Repository](https://github.com/georgw777/chesscog)
- **Description:** Dataset for chess board and piece recognition
- **Extract to:** `data/chesscog/`

## Important Notes

- Do not commit the actual dataset files to the repository
- The `.gitignore` file is configured to exclude:
  - `data/*.zip`
  - `data/*.tar.gz`
  - `data/ChessReD/`
  - `data/chesscog/`
- Make sure to download and extract datasets before running the pipeline
