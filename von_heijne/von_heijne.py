#!/usr/bin/env python
# coding: utf-8

import os
import math
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio import SeqIO
from sklearn.metrics import precision_recall_curve, f1_score, matthews_corrcoef

# Constants
AMINO_ACIDS = [
    'G', 'A', 'V', 'P', 'L', 'I', 'M', 'F', 'W', 'Y',
    'S', 'T', 'C', 'N', 'Q', 'H', 'D', 'E', 'K', 'R'
]

AMINO_ACIDS_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

SWISSPROT_FREQ = {
    'A': 8.25, 'Q': 3.93, 'L': 9.64, 'S': 6.65,
    'R': 5.52, 'E': 6.71, 'K': 5.80, 'T': 5.36,
    'N': 4.06, 'G': 7.07, 'M': 2.41, 'W': 1.10,
    'D': 5.46, 'H': 2.27, 'F': 3.86, 'Y': 2.92,
    'C': 1.38, 'I': 5.91, 'P': 4.74, 'V': 6.85
}

# Normalize SwissProt frequencies
SWISSPROT_FREQ_NORMALIZED = {k: round(v / 100, 3) for k, v in SWISSPROT_FREQ.items()}

# Base directory path
#BASE_PATH = "/Users/gianlucapiccolo/Desktop/lab2_2024/script/files/"

# This will give you the path to the directory just above the current directory
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)

    # If you want to navigate into the "files" directory in the parent directory:
base_path_url = os.path.join(parent_dir_path, "data_fetch/files/")

    # You can also normalize the path using os.path.abspath to get the full path:
#parent_dir_path = os.path.abspath(parent_dir_path)
BASE_PATH = os.path.abspath(base_path_url)

def filter_dataframe_by_dict_keys(df, dict_keys, column_name="ID"):
    """
    Filters a dataframe to keep only the rows where the values in the specified column are present in the dict keys.

    Parameters:
        df (pd.DataFrame): The dataframe to filter.
        dict_keys (iterable): The keys of the dictionary.
        column_name (str): The column name in the dataframe to check.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    return df[df[column_name].isin(dict_keys)]

def load_sequences(fasta_file):
    """
    Loads sequences from a FASTA file into a dictionary.

    Parameters:
        fasta_file (str): Path to the FASTA file.

    Returns:
        dict: A dictionary mapping sequence IDs to sequences.
    """
    return {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}

def load_data(base_path):
    """
    Loads all necessary data from the base path.

    Parameters:
        base_path (str): The base directory path where files are located.

    Returns:
        tuple: sequences_dict_p, sequences_dict_n, train_pos_df, train_neg_df, bench_pos_df, bench_neg_df, Training_df, Benchmark_df
    """
    # Load positive and negative sequences
    fasta_file_p = os.path.join(base_path, 'cluster-results_pos_rep_seq.fasta')
    sequences_dict_p = load_sequences(fasta_file_p)

    fasta_file_n = os.path.join(base_path, 'cluster-results_neg_rep_seq.fasta')
    sequences_dict_n = load_sequences(fasta_file_n)

    # Load TSV files
    pos_tsv = os.path.join(base_path, 'split_pos.tsv')
    neg_tsv = os.path.join(base_path, 'split_neg.tsv')

    positivi_df = pd.read_csv(pos_tsv, sep="\t")
    negativi_df = pd.read_csv(neg_tsv, sep="\t")

    # Filter positive dataframe based on available sequences
    filtered_pos_df = filter_dataframe_by_dict_keys(positivi_df, sequences_dict_p.keys())
    
    # Split into training and benchmark sets
    train_pos_df = filtered_pos_df[filtered_pos_df["set"] == "T"].copy()
    bench_pos_df = filtered_pos_df[filtered_pos_df["set"] == "B"].copy()

    train_neg_df = negativi_df[negativi_df["set"] == "T"].copy()
    bench_neg_df = negativi_df[negativi_df["set"] == "B"].copy()

    # Assign labels
    train_pos_df['label'] = '+'
    train_neg_df['label'] = '-'
    bench_pos_df['label'] = '+'
    bench_neg_df['label'] = '-'

    # Combine positive and negative dataframes
    Training_df = pd.concat([train_pos_df, train_neg_df], axis=0).reset_index(drop=True)
    Benchmark_df = pd.concat([bench_pos_df, bench_neg_df], axis=0).reset_index(drop=True)

    return sequences_dict_p, sequences_dict_n, train_pos_df, train_neg_df, bench_pos_df, bench_neg_df, Training_df, Benchmark_df

def get_split(df, iteration, n_folds=5):
    """
    Splits the dataframe into test, validation, and training sets based on fold assignment.

    Parameters:
        df (pd.DataFrame): The dataframe to split.
        iteration (int): Current iteration number.
        n_folds (int): Total number of folds.

    Returns:
        tuple: (test_set, validation_set, training_set)
    """
    test_fold = iteration % n_folds
    validation_fold = (iteration + 1) % n_folds
    training_folds = [(iteration + i) % n_folds for i in range(2, n_folds)]

    test_set = df[df['Fold'] == test_fold].reset_index(drop=True)
    validation_set = df[df['Fold'] == validation_fold].reset_index(drop=True)
    training_set = df[(df['Fold'].isin(training_folds)) & (df['label'] == '+')].reset_index(drop=True)

    return test_set, validation_set, training_set

def extract_training_sequences(training_df, sequences_dict_p):
    """
    Extracts training sequences based on cleavage positions.

    Parameters:
        training_df (pd.DataFrame): DataFrame containing training data with 'ID' and 'Cleavage_pos'.
        sequences_dict_p (dict): Dictionary mapping sequence IDs to sequences.

    Returns:
        list: List of extracted sequences.
    """
    extracted_sequences = []
    for _, row in training_df.iterrows():
        sequence_id = row['ID']
        cleavage_pos = row['Cleavage_pos']

        sequence = sequences_dict_p.get(sequence_id)
        if sequence:
            start_pos = max(0, cleavage_pos - 13)
            end_pos = min(len(sequence), cleavage_pos + 2)
            extracted_sequence = sequence[start_pos:end_pos]
            extracted_sequences.append(extracted_sequence)
        else:
            print(f"Sequence ID {sequence_id} not found in sequences_dict_p.")
    return extracted_sequences

def extract_whole_seq(df, sequences_dict_n, sequences_dict_p):
    """
    Extracts entire sequences from the dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame containing the sequences.
        sequences_dict_n (dict): Dictionary for negative sequences.
        sequences_dict_p (dict): Dictionary for positive sequences.

    Returns:
        list: List of tuples (sequence, label)
    """
    extracted_sequences = []
    for _, row in df.iterrows():
        sequence_id = row['ID']
        label = row["label"]

        if label == "+":
            sequence = sequences_dict_p.get(sequence_id)
            if sequence is None:
                print(f"Error: Sequence ID {sequence_id} with label '+' not found.")
        else:
            sequence = sequences_dict_n.get(sequence_id)
            if sequence is None:
                print(f"Error: Sequence ID {sequence_id} with label '-' not found.")

        if sequence:
            extracted_sequences.append((sequence, label))
    return extracted_sequences

def one_hot_encode(sequence, amminoacidi_index, seq_length=15, num_amminoacidi=20):
    """
    One-hot encodes a given sequence, counting occurrences per position.

    Parameters:
        sequence (str): Amino acid sequence.
        amminoacidi_index (dict): Dictionary mapping amino acids to indices.
        seq_length (int): Length of the sequence to consider.
        num_amminoacidi (int): Number of amino acids.

    Returns:
        np.ndarray: One-hot encoded matrix with counts.
    """
    one_hot_matrix = np.zeros((seq_length, num_amminoacidi))
    for i, aa in enumerate(sequence[:seq_length]):
        if aa in amminoacidi_index:
            aa_idx = amminoacidi_index[aa]
            one_hot_matrix[i, aa_idx] += 1
    return one_hot_matrix

def create_matrix(sequences, amminoacidi_index, swissprot_freq_normalized, seq_length=15, num_amminoacidi=20):
    """
    Creates a log-normalized matrix based on sequence occurrences and SwissProt frequencies.

    Parameters:
        sequences (list): List of sequences.
        amminoacidi_index (dict): Dictionary mapping amino acids to indices.
        swissprot_freq_normalized (dict): Normalized SwissProt frequencies.
        seq_length (int): Length of the sequences to consider.
        num_amminoacidi (int): Number of amino acids.

    Returns:
        np.ndarray: Log-normalized matrix.
    """
    # Initialize matrix with ones to avoid division by zero
    matrix_amminoacidi = np.ones((seq_length, num_amminoacidi))
    
    for sequence in sequences:
        one_hot = one_hot_encode(sequence, amminoacidi_index, seq_length, num_amminoacidi)
        matrix_amminoacidi += one_hot
    
    matrix_divided = matrix_amminoacidi / (len(sequences) + 20)  # Avoid division by zero
    for aa, idx in amminoacidi_index.items():
        matrix_divided[:, idx] /= swissprot_freq_normalized.get(aa, 1)  # Avoid KeyError
    matrix_log = np.log(matrix_divided)
    
    return matrix_log

def get_score(matrix, sequences, amminoacidi_index, window_size=15):
    """
    Computes the maximum score for each sequence based on the scoring matrix.

    Parameters:
        matrix (np.ndarray): Scoring matrix.
        sequences (list): List of tuples (sequence, label).
        amminoacidi_index (dict): Dictionary mapping amino acids to indices.
        window_size (int): Size of the window to slide.

    Returns:
        list: List of tuples (max_score, label).
    """
    results = []
    for sequence, label in sequences:
        max_score = -math.inf  # Initialize to negative infinity
        seq_len = len(sequence)

        for i in range(seq_len - window_size + 1):
            window = sequence[i:i + window_size]
            score = 0
            for pos, aa in enumerate(window):
                if aa in amminoacidi_index:
                    aa_idx = amminoacidi_index[aa]
                    score += matrix[pos][aa_idx]
            if score > max_score:
                max_score = score
        results.append((max_score, label))
    return results

def calculate_mcc(TP, FP, TN, FN):
    """
    Calculates the Matthews Correlation Coefficient (MCC).

    Parameters:
        TP (int): True Positives.
        FP (int): False Positives.
        TN (int): True Negatives.
        FN (int): False Negatives.

    Returns:
        float: MCC value.
    """
    numerator = (TP * TN) - (FP * FN)
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denominator == 0:
        return 0
    return numerator / denominator

def find_best_threshold(val_predictions, threshold_list):
    """
    Finds the best threshold based on MCC.

    Parameters:
        val_predictions (list): List of tuples (score, label).
        threshold_list (list): List of thresholds to evaluate.

    Returns:
        tuple: (best_threshold, max_MCC, y_real, y_score)
    """
    best_th = 0
    max_MCC = -1
    y_real = []
    y_score = []

    for th in threshold_list:
        TP = FP = TN = FN = 0
        for score, label in val_predictions:
            y_score.append(score)
            y_real.append(1 if label == "+" else 0)
            pred = "+" if score > th else "-"
            if label == "+" and pred == "+":
                TP += 1
            elif label == "-" and pred == "-":
                TN += 1
            elif label == "+" and pred == "-":
                FN += 1
            else:
                FP += 1
        mcc = calculate_mcc(TP, FP, TN, FN)
        if mcc > max_MCC:
            max_MCC = mcc
            best_th = th
    # After loop, assign y_real and y_score correctly
    y_real = [1 if label == "+" else 0 for _, label in val_predictions]
    y_score = [score for score, _ in val_predictions]
    return best_th, max_MCC, y_real, y_score

def test_performance(test_predictions, threshold):
    """
    Tests performance by computing MCC on test set based on the given threshold.

    Parameters:
        test_predictions (list): List of tuples (score, label).
        threshold (float): Threshold to apply.

    Returns:
        float: MCC score on test set.
    """
    TP = FP = TN = FN = 0
    for score, label in test_predictions:
        pred = "+" if score > threshold else "-"
        if label == "+" and pred == "+":
            TP += 1
        elif label == "-" and pred == "-":
            TN += 1
        elif label == "+" and pred == "-":
            FN += 1
        else:
            FP += 1
    return calculate_mcc(TP, FP, TN, FN)

def plot_kingdom_distribution(df: pd.DataFrame, title: str):
    """
    Plot the distribution of kingdoms in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing a 'Kingdom' column.
        title (str): Title for the plot.
    """
    kingdom_counts = df['Kingdom'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(kingdom_counts, labels=kingdom_counts.index, autopct='%1.1f%%',
            startangle=90, colors=['skyblue', 'green', 'purple', 'cyan'])
    plt.title(title)
    plt.show()

def extract_and_save_sequences(fasta_path, ids_to_extract, output_fasta):
    """
    Estrae le sequenze da un file FASTA basandosi su una lista di ID e le salva in un nuovo file FASTA.
    
    Parameters:
        fasta_path (str): Percorso al file FASTA di input ("cluster.fasta").
        ids_to_extract (set): Set di ID da estrarre.
        output_fasta (str): Percorso al file FASTA di output dove salvare le sequenze estratte.
    """
    # Crea un indice del file FASTA per accesso rapido
    fasta_index = SeqIO.index(fasta_path, "fasta")
    
    # Lista per memorizzare le sequenze estratte
    extracted_sequences = []
    found_ids = set()
    
    for seq_id in ids_to_extract:
        if seq_id in fasta_index:
            extracted_sequences.append(fasta_index[seq_id])
            found_ids.add(seq_id)
        else:
            print(f"Warning: ID {seq_id} non trovato in {fasta_path}")
    
    # Chiudi l'indice
    fasta_index.close()
    
    # Salva le sequenze estratte nel nuovo file FASTA
    if extracted_sequences:
        SeqIO.write(extracted_sequences, output_fasta, "fasta")
        print(f"{len(extracted_sequences)} sequenze estratte e salvate in '{output_fasta}'")
    else:
        print("Nessuna sequenza è stata estratta. Verifica gli ID forniti.")


def main():
    # Load data
    sequences_dict_p, sequences_dict_n, train_pos_df, train_neg_df, bench_pos_df, bench_neg_df, Training_df, Benchmark_df = load_data(BASE_PATH)

    # Cross-validation parameters
    n_folds = 5
    th_list = [round(x, 1) for x in np.arange(5.0, 12.0, 0.1)]

    best_thresholds = []
    results = []
    fold_index = 1

    for iteration in range(n_folds):
        print(f"Cross Validation: Fold {fold_index}")
        
        # Split data
        test_set, validation_set, training_set = get_split(Training_df, iteration, n_folds=n_folds)
        
        # Extract training sequences and create matrix
        training_sequences = extract_training_sequences(training_set, sequences_dict_p)
        scoring_matrix = create_matrix(training_sequences, AMINO_ACIDS_INDEX, SWISSPROT_FREQ_NORMALIZED)
        
        # Extract validation and test sequences
        validation_sequences = extract_whole_seq(validation_set, sequences_dict_n, sequences_dict_p)
        test_sequences = extract_whole_seq(test_set, sequences_dict_n, sequences_dict_p)
        
        # Get predictions
        val_predictions = get_score(scoring_matrix, validation_sequences, AMINO_ACIDS_INDEX)
        test_predictions = get_score(scoring_matrix, test_sequences, AMINO_ACIDS_INDEX)
        
        # Find the best threshold based on MCC
        best_th, max_MCC, y_real, y_score = find_best_threshold(val_predictions, th_list)
        
        # Compute test MCC with best threshold
        final_MCC = test_performance(test_predictions, best_th)
        
        # Compute Precision-Recall curve
        precision, recall, thresholds_pr = precision_recall_curve(y_real, y_score)
        
        # Compute F1 scores for each threshold in th_list
        f1_scores = []
        optimal_f1_threshold = th_list[0]
        max_f1 = 0
        for th in th_list:
            y_pred = (np.array(y_score) >= th).astype(int)
            current_f1 = f1_score(y_real, y_pred)
            f1_scores.append(current_f1)
            if current_f1 > max_f1:
                max_f1 = current_f1
                optimal_f1_threshold = th
        
        # Compute test MCC using optimal F1 threshold
        test_mcc_f1 = test_performance(test_predictions, optimal_f1_threshold)
        
        # Find max MCC across thresholds
        mcc_scores = []
        for th in th_list:
            y_pred_val = (np.array(y_score) >= th).astype(int)
            mcc = matthews_corrcoef(y_real, y_pred_val)
            mcc_scores.append(mcc)
        max_mcc = np.max(mcc_scores)
        max_mcc_threshold = th_list[np.argmax(mcc_scores)]
        
        # Append best threshold
        best_thresholds.append(best_th)
        
        # Plot Precision-Recall curve
        plt.figure()
        plt.plot(recall, precision, marker=".")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (Fold {fold_index})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Print iteration results
        print(f"The best threshold (MCC) is: {best_th} with MCC: {max_MCC}")
        print(f"The maximum MCC is: {max_MCC} at threshold: {max_mcc_threshold}")
        print(f"Optimal Threshold for F1 is: {optimal_f1_threshold} with F1 Score: {max_f1}")
        print(f"Performance on the test set, MCC (MCC Threshold): {final_MCC}")
        print(f"Performance on the test set, MCC (F1 Threshold): {test_mcc_f1}\n")
        
        # Store results
        results.append({
            "Fold": fold_index,
            "Best Threshold MCC": best_th,
            "Max MCC": max_MCC,
            "Max MCC Threshold": max_mcc_threshold,
            "Optimal Threshold F1": optimal_f1_threshold,
            "Max F1 Score": max_f1,
            "Test MCC (MCC Threshold)": final_MCC,
            "Test MCC (F1 Threshold)": test_mcc_f1
        })
        
        fold_index += 1
    
    # Calculate the average threshold
    th_average = np.mean(best_thresholds)
    print(f"The average threshold among the {n_folds} iterations is: {th_average}")
    
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    
    # Save the results to a CSV file
    results_csv_path = os.path.join(BASE_PATH, "von_heijne_cross_validation_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results have been saved to '{results_csv_path}'")
    
    # Evaluate on Benchmark set using the average threshold
    print("\nEvaluating on Benchmark set...")
    
    # Create training sequences from all positive training data
    training_pos_sequences = extract_training_sequences(train_pos_df, sequences_dict_p)
    matrix = create_matrix(training_pos_sequences, AMINO_ACIDS_INDEX, SWISSPROT_FREQ_NORMALIZED)
    
    # Extract benchmark sequences
    benchmark_sequences = extract_whole_seq(Benchmark_df, sequences_dict_n, sequences_dict_p)
    bench_predictions = get_score(matrix, benchmark_sequences, AMINO_ACIDS_INDEX)

    #print(Benchmark_df)
    #print(bench_predictions)
    
    # Calculate MCC on Benchmark set
    final_MCC = test_performance(bench_predictions, th_average)
    print(f"Benchmark set MCC with average threshold {th_average}: {final_MCC}")


    # === ESTRAGGO FALSE POSITIVE E FALSE NEGATIVE ===
    # Calcola la soglia media (th_average)
    th_average = np.mean(best_thresholds)

    # Estrai le etichette predette basate sulla soglia media
    pred_labels = ['+' if score > th_average else '-' for score, _ in bench_predictions]

    # Assicurati che la lunghezza delle predizioni corrisponda al dataframe
    if len(pred_labels) != len(Benchmark_df):
        raise ValueError("Il numero di predizioni non corrisponde al numero di righe del dataframe.")

    # Aggiungi le predizioni al dataframe
    Benchmark_df['pred_label'] = pred_labels

    # Identifica False Positives e False Negatives
    Benchmark_df['False_Positive'] = (Benchmark_df['pred_label'] == '+') & (Benchmark_df['label'] == '-')
    Benchmark_df['False_Negative'] = (Benchmark_df['pred_label'] == '-') & (Benchmark_df['label'] == '+')

    # Estrai i False Positives e False Negatives
    false_positives = Benchmark_df[Benchmark_df['False_Positive']]
    false_negatives = Benchmark_df[Benchmark_df['False_Negative']]

    plot_kingdom_distribution(Benchmark_df, 'Distribution of Kingdoms (Benchmark)')
    plot_kingdom_distribution(false_positives, 'Distribution of Kingdoms (False Positives)')
    plot_kingdom_distribution(false_negatives, 'Distribution of Kingdoms (False Negatives)')

    # Visualizza i risultati
    #print("False Positives:")
    #print(false_positives)

    #print("\nFalse Negatives:")
    #print(false_negatives)

    # === FALSE POSITIVE ANALYSIS === #
    FP_list = false_positives["ID"].to_list()

    FP_TM = false_positives["Cleavage_pos"].sum()
    Neg_TM = bench_neg_df["Cleavage_pos"].sum()
    FPR_TM = FP_TM / Neg_TM if Neg_TM > 0 else 0
    print(f"FP_TM: {FP_TM}")
    print(f"Neg_TM: {Neg_TM}")
    print(f"The fraction of negatives having the transmembrane misclassified: {FPR_TM:.4f}")
    print(f"The percentage of False Positives having the transmembrane domain: {FP_TM / len(FP_list) if len(FP_list) > 0 else 0:.4f}")


    # === FALSE NEGATIVE ANALYSIS === 
    # compute logo of false negatives
    # Estrai gli ID dei False Negatives
    ids_false_negatives = set(false_negatives['ID'].str.upper())  # Assicurati che gli ID siano in maiuscolo

    # Percorso al file "cluster.fasta"
    cluster_fasta_path = os.path.join(BASE_PATH, "cluster-results_pos_rep_seq.fasta")
    
    # Definisci il percorso per il nuovo file FASTA
    output_fasta_path = os.path.join(BASE_PATH, "von_heijne_false_negatives_sequences_per_logo.fasta")
    
    # Chiama la funzione per estrarre e salvare le sequenze
    extract_and_save_sequences(
        fasta_path=cluster_fasta_path,
        ids_to_extract=ids_false_negatives,
        output_fasta=output_fasta_path
    )
    
if __name__ == "__main__":
    main()