import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from itertools import chain
from config import VALID_LABELS

def evaluate_subtask_1(gold_filepath, pred_filepath):
    gold = pd.read_csv(gold_filepath, sep="\t", header=None, names=["doc_id", "start", "end", "flag"])
    pred = pd.read_csv(pred_filepath, sep="\t", header=None, names=["doc_id", "start", "end", "flag"])

    # Handle NaN values by replacing them with False
    gold["flag"] = gold["flag"].fillna(False)
    pred["flag"] = pred["flag"].fillna(False)

    # Check if document IDs and offsets match
    if not gold[["doc_id", "start", "end"]].equals(pred[["doc_id", "start", "end"]]):
        raise ValueError("Mismatch in document IDs or offsets between gold and prediction files")
    
    # Compute evaluation metrics
    accuracy = accuracy_score(gold["flag"], pred["flag"])
    precision = precision_score(gold["flag"], pred["flag"], pos_label=True)
    recall = recall_score(gold["flag"], pred["flag"], pos_label=True)
    f1 = f1_score(gold["flag"], pred["flag"], pos_label=True)
    
    print("Subtask 1 Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def load_subtask_2_data(filepath):
    """Loads a Subtask 2 file where the number of columns varies (labels are optional)."""
    with open(filepath, "r", encoding="utf-8") as file:
        lines = [line.strip().split("\t") for line in file]

    # Convert to DataFrame
    df = pd.DataFrame(lines)

    # Ensure at least 3 required columns exist
    if df.shape[1] < 3:
        raise ValueError(f"File {filepath} has fewer than 3 columns in some rows!")

    # Rename first three columns, remaining ones are labels
    df.rename(columns={0: "doc_id", 1: "start", 2: "end"}, inplace=True)

    # Convert start & end offsets to integers
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    # Store persuasion technique labels as a list
    label_columns = df.columns[3:]  # Extra columns
    df["labels"] = df[label_columns].apply(lambda x: [label for label in x if label in VALID_LABELS], axis=1)

    # Drop extra columns since labels are now stored in a list
    df = df[["doc_id", "start", "end", "labels"]]

    return df

def evaluate_subtask_2(gold_filepath, pred_filepath, per_class_results=False):
    """Evaluates Subtask 2 predictions using micro and macro F1 scores, after format checking."""
    
    # Load validated data
    gold = load_subtask_2_data(gold_filepath)
    pred = load_subtask_2_data(pred_filepath)

    # Sort both DataFrames to ensure correct alignment
    gold = gold.sort_values(by=["doc_id", "start", "end"]).reset_index(drop=True)
    pred = pred.sort_values(by=["doc_id", "start", "end"]).reset_index(drop=True)

    # Ensure document IDs and offsets match
    if not gold[["doc_id", "start", "end"]].equals(pred[["doc_id", "start", "end"]]):
        raise ValueError("Mismatch in document IDs or offsets between gold and prediction files")

    # Convert label lists to sets for evaluation
    gold_labels = gold["labels"].apply(set)
    pred_labels = pred["labels"].apply(set)

    # valid_classes = sorted(set(chain.from_iterable(gold_labels)) | set(chain.from_iterable(pred_labels)))
    valid_classes = sorted(set(chain.from_iterable(gold_labels)))

    # Convert lists of labels to binary multi-label format for F1 computation
    def multilabel_binarize(label_sets, valid_labels):
        # Create a binary matrix
        return np.array([[1 if label in row else 0 for label in valid_labels] for row in label_sets])

    # Binarize the labels
    y_true = multilabel_binarize(gold_labels, valid_classes)
    y_pred = multilabel_binarize(pred_labels, valid_classes)

    # Compute evaluation metrics
    # Calculate Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    micro_precision = precision_score(y_true, y_pred, average="micro")
    micro_recall = recall_score(y_true, y_pred, average="micro")
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Print results
    print("Subtask 2 Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    if per_class_results:
        # Get results per class
        class_results = {}
        for i, class_name in enumerate(valid_classes):
            # Extract the binary vectors for the current class
            y_true_class = y_true[:, i]
            y_pred_class = y_pred[:, i]

            accuracy_class = accuracy_score(y_true_class, y_pred_class)
            precision_class = precision_score(y_true_class, y_pred_class,
                                            zero_division=0)
            recall_class = recall_score(y_true_class, y_pred_class,
                                        zero_division=0)
            f1_class = f1_score(y_true_class, y_pred_class, zero_division=0)

            # Count statistics for gold and pred
            gold_count = np.sum(y_true[:, i])
            pred_count = np.sum(y_pred[:, i])

            class_results[class_name] = {
                "Gold Count": gold_count,
                "Pred Count": pred_count,
                "Accuracy": accuracy_class,
                "Precision": precision_class,
                "Recall": recall_class,
                "F1 Score": f1_class
            }

        # Create DataFrame from class results
        df_class_results = pd.DataFrame(class_results).T

        # Create DataFrame for overall metrics
        overall_metrics = pd.DataFrame({
            "Overall-Micro": [accuracy, micro_precision, micro_recall, micro_f1],
            "Overall-Macro": [None, macro_precision, macro_recall, macro_f1]
        }, index=["Accuracy", "Precision", "Recall", "F1 Score"]).T

        # Concatenate DataFrames
        df_class_results = pd.concat([df_class_results, overall_metrics], axis=0)

        # Save to file
        output_filepath = "subtask_2_per_class_evaluation.csv"
        df_class_results.to_csv(output_filepath)
        print(f"\nPer-class evaluation results saved to {output_filepath}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate submission files for different subtasks.")
    parser.add_argument("subtask", choices=["subtask1", "subtask2"], help="Which subtask to evaluate")
    parser.add_argument("gold_filepath", type=str, help="Path to the gold standard file")
    parser.add_argument("pred_filepath", type=str, help="Path to the predictions file")

    # Only add the per_class_results argument for subtask 2
    parser.add_argument("--per_class_results", action="store_true", help="Save per-class results (only for Subtask 2)")

    args = parser.parse_args()

    if args.subtask == "subtask1":
        evaluate_subtask_1(args.gold_filepath, args.pred_filepath)
    else:
        evaluate_subtask_2(args.gold_filepath, args.pred_filepath, args.per_class_results)

if __name__ == "__main__":
    main()