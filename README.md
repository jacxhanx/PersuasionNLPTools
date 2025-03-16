# 🛠️ PersuasionNLPTools

## Overview  
This repository provides tools for checking the format conformity of submissions and evaluating predictions against gold-standard labels for the [Shared Task on the Detection and Classification of Persuasion Techniques in Texts for Slavic Languages](https://bsnlp.cs.helsinki.fi/shared-task.html), part of the [SNLP 2025 Workshop](https://bsnlp.cs.helsinki.fi/index.html) that is going to be held in conjunction with the [ACL 2025 Conference](https://2025.aclweb.org/).

The tools in this repository are provided to support task participants in verifying submission formats and evaluating the detection and classification of persuasion techniques for this shared task.


It includes two main scripts:
- `conformity_checker.py` – Ensures submission files are correctly formatted.
- `submission_evaluator.py` – Compares submitted predictions with the gold standard.

---

## 🏗️ Installation

1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/PersuasionNLPTools.git
   cd PersuasionNLPTools

2. (Optional) Create a virtual environment and install dependencies
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    pip install -r requirements.txt


## ▶️ Usage

1. Format Checking: conformity_checker.py

    This script ensures that a submission file meets the required format for a given subtask.

    Usage:
    ```bash
    python conformity_checker.py <subtask> <submission_file>

Arguments:
* subtask: Choose "subtask1" or "subtask2".
* submission_file: Path to the submission file.

    Example:
    ```bash
    python conformity_checker.py subtask1 submission.csv

    
2. Evaluation: submission_evaluator.py

    This script compares the submitted predictions against the gold standard to compute evaluation metrics.
    
    Usage:
    ```bash
    python submission_evaluator.py <subtask> <gold_filepath> <pred_filepath> [--per_class_results]

Arguments:
* subtask: Choose "subtask1" or "subtask2".
* gold_filepath: Path to the gold standard labels file.
* pred_filepath: Path to the predictions file.
* --per_class_results (optional): If specified, saves to file detailed per-class results (only for Subtask 2).

    Examples:
    ```bash
    # Evaluate Subtask 1
    python submission_evaluator.py subtask1 gold.csv pred.csv

    # Evaluate Subtask 2 (default)
    python submission_evaluator.py subtask2 gold.csv pred.csv

    # Evaluate Subtask 2 with per-class results
    python submission_evaluator.py subtask2 gold.csv pred.csv --per_class_results


## 📌 Notes

* Ensure your submission files are formatted correctly before evaluation.
* For Subtask 2, use --per_class_results if you need per-class metrics.


## 📩 Contact

For issues or questions, please open an issue on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/jacxhanx/PersuasionNLPTools/blob/main/LICENSE) file for details.