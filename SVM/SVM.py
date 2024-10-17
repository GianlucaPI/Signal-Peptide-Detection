#!/usr/bin/env python
# coding: utf-8

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import combinations
from datetime import datetime

from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import ProtParamData

from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    matthews_corrcoef, f1_score, precision_score, recall_score,
    precision_recall_curve
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split

# Constants
AMINO_ACIDS = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
              "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

CHOU_FASMAN = {
    'A': 1.420, 'C': 0.700, 'D': 1.010, 'E': 1.510, 'F': 1.130,
    'G': 0.570, 'H': 1.000, 'I': 1.080, 'K': 1.160, 'L': 1.210,
    'M': 1.450, 'N': 0.670, 'P': 0.570, 'Q': 1.110, 'R': 0.980,
    'S': 0.770, 'T': 0.830, 'V': 1.060, 'W': 1.080, 'Y': 0.690
}

ZHAO_LONDON = {
    'A': 0.380, 'C': -0.300, 'D': -3.270, 'E': -2.900, 'F': 1.980,
    'G': -0.190, 'H': -1.440, 'I': 1.970, 'K': -3.460, 'L': 1.820,
    'M': 1.400, 'N': -1.620, 'P': -1.440, 'Q': -1.840, 'R': -2.570,
    'S': -0.530, 'T': -0.320, 'V': 1.460, 'W': 1.530, 'Y': 0.490
}

AMINO_ACID_CHARGE = {
    'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0, 'G': 0, 'H': 1,
    'I': 0, 'K': 1, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0,
    'R': 1, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

# Base directory path
#BASE_PATH = "/Users/gianlucapiccolo/Desktop/lab2_2024/script/files/"

# This will give you the path to the directory just above the current directory
parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)

    # If you want to navigate into the "files" directory in the parent directory:
base_path_url = os.path.join(parent_dir_path, "data_fetch/files/")

    # You can also normalize the path using os.path.abspath to get the full path:
#parent_dir_path = os.path.abspath(parent_dir_path)
BASE_PATH = os.path.abspath(base_path_url) 

def get_sequence_composition(sequence: str) -> list:
    """
    Calculate the percentage composition of each amino acid in the sequence.
    
    Parameters:
        sequence (str): Amino acid sequence.
        
    Returns:
        list: Percentage of each amino acid in the sequence.
    """
    aa_percentage = []
    invalid_residues = sum(1 for residue in sequence if residue not in AMINO_ACIDS)
    valid_length = len(sequence) - invalid_residues if len(sequence) > invalid_residues else 1  # Avoid division by zero

    for residue in AMINO_ACIDS:
        count = sequence.count(residue)
        percentage = count / valid_length
        aa_percentage.append(percentage)
    
    return aa_percentage

def load_fasta_sequences(fasta_path: str) -> dict:
    """
    Load sequences from a FASTA file into a dictionary.
    
    Parameters:
        fasta_path (str): Path to the FASTA file.
        
    Returns:
        dict: Mapping from sequence ID to sequence string.
    """
    return {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, 'fasta')}

def get_dataframe(base_path: str, split_tsv_path: str, fasta_path: str, class_label: int) -> pd.DataFrame:
    """
    Load TSV and FASTA files, and create a DataFrame with sequences.
    
    Parameters:
        base_path (str): Base directory path.
        split_tsv_path (str): Path to the split TSV file.
        fasta_path (str): Path to the FASTA file.
        class_label (int): Class label (1 for positive, 0 for negative).
        
    Returns:
        pd.DataFrame: DataFrame with sequence information and features.
    """
    df = pd.read_csv(os.path.join(base_path, split_tsv_path), sep="\t")
    seq_dict = load_fasta_sequences(os.path.join(base_path, fasta_path))
    
    sequences = []
    for _, row in df.iterrows():
        seq_id = row["ID"]
        sequence = seq_dict.get(seq_id, "")
        sequences.append(sequence[:40] if sequence else "")
    
    df['Class'] = class_label
    df['First_40_AAs'] = sequences
    
    return df

def get_hydropathy(sequence: str, window_size: int = 5) -> tuple:
    """
    Calculate hydropathy features using ProteinAnalysis.
    
    Parameters:
        sequence (str): Amino acid sequence.
        window_size (int): Window size for hydropathy calculation.
        
    Returns:
        tuple: Maximum and average hydropathy values.
    """
    padding = "X" * (window_size // 2)
    padded_sequence = padding + sequence + padding
    analysis = ProteinAnalysis(padded_sequence)
    hydropathy = analysis.protein_scale(ProtParamData.kd, window_size)
    
    max_hydropathy = max(hydropathy)
    avg_hydropathy = np.mean(hydropathy)
    
    return max_hydropathy, avg_hydropathy

def get_charge(sequence: str, window_size: int = 3) -> tuple:
    """
    Calculate charge features using ProteinAnalysis.
    
    Parameters:
        sequence (str): Amino acid sequence.
        window_size (int): Window size for charge calculation.
        
    Returns:
        tuple: Maximum charge and normalized charge position.
    """
    padding = "X" * (window_size // 2)
    padded_sequence = padding + sequence + padding
    analysis = ProteinAnalysis(padded_sequence)
    charge = analysis.protein_scale(AMINO_ACID_CHARGE, window_size)
    
    if not charge:
        return 0.0, 0.0
    
    max_charge = max(charge)
    try:
        max_charge_pos = charge.index(max_charge)
    except ValueError:
        max_charge_pos = 0
    
    normalized_charge = max_charge_pos / len(sequence) if len(sequence) > 0 else 0.0
    norm_corrected = 1 - normalized_charge
    
    return max_charge, norm_corrected

def get_secondary_structure_propensity(sequence: str, window_size: int = 7) -> tuple:
    """
    Calculate secondary structure propensity using ProteinAnalysis.
    
    Parameters:
        sequence (str): Amino acid sequence.
        window_size (int): Window size for propensity calculation.
        
    Returns:
        tuple: Maximum and average alpha-helix propensity.
    """
    padding = "X" * (window_size // 2)
    padded_sequence = padding + sequence + padding
    analysis = ProteinAnalysis(padded_sequence)
    helix_propensity = analysis.protein_scale(CHOU_FASMAN, window_size)
    
    max_propensity = max(helix_propensity)
    avg_propensity = np.mean(helix_propensity)
    
    return max_propensity, avg_propensity

def get_transmembrane_propensity(sequence: str, window_size: int = 7) -> tuple:
    """
    Calculate transmembrane propensity using ProteinAnalysis.
    
    Parameters:
        sequence (str): Amino acid sequence.
        window_size (int): Window size for propensity calculation.
        
    Returns:
        tuple: Maximum and average transmembrane propensity.
    """
    padding = "X" * (window_size // 2)
    padded_sequence = padding + sequence + padding
    analysis = ProteinAnalysis(padded_sequence)
    transmembrane_propensity = analysis.protein_scale(ZHAO_LONDON, window=window_size)
    
    max_trans_propensity = max(transmembrane_propensity)
    avg_trans_propensity = np.mean(transmembrane_propensity)
    
    return max_trans_propensity, avg_trans_propensity

def split_data(df: pd.DataFrame, iteration: int, n_folds: int = 5) -> tuple:
    """
    Split the DataFrame into test, validation, and training sets based on fold assignment.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to split.
        iteration (int): Current iteration index.
        n_folds (int): Number of folds for cross-validation.
        
    Returns:
        tuple: Test set, validation set, training set, test_fold, validation_fold, training_folds
    """
    test_fold = iteration % n_folds
    validation_fold = (iteration + 1) % n_folds
    training_folds = [(iteration + i) % n_folds for i in range(2, n_folds)]
    
    test_set = df[df['Fold'] == test_fold].reset_index(drop=True)
    validation_set = df[df['Fold'] == validation_fold].reset_index(drop=True)
    training_set = df[df['Fold'].isin(training_folds)].reset_index(drop=True)
    
    return test_set, validation_set, training_set, test_fold, validation_fold, training_folds

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from the DataFrame sequences.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing sequences and labels.
        
    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    features_data = []
    
    for _, row in df.iterrows():
        seq_id = row['ID']
        sequence = row['First_40_AAs']
        class_label = row['Class']
        
        # 1. Amino acid composition up to 22 residues
        aa_percentage = get_sequence_composition(sequence[:22])
        
        # 2. Hydropathy features
        max_kd, avg_kd = get_hydropathy(sequence)
        
        # 3. Charge features
        max_charge, norm_charge_pos = get_charge(sequence)
        
        # 4. Secondary structure propensity
        max_helix_prop, avg_helix_prop = get_secondary_structure_propensity(sequence)
        
        # 5. Transmembrane propensity
        max_trans_prop, avg_trans_prop = get_transmembrane_propensity(sequence)
        
        feature_dict = {
            'ID': seq_id,
            'Class': class_label,
            **{f'AA_Percentage_{aa}': aa_percentage[i] for i, aa in enumerate(AMINO_ACIDS)},
            'Max_KD': max_kd,
            'Avg_KD': avg_kd,
            'Max_Charge': max_charge,
            'Normalized_Charge_Position': norm_charge_pos,
            'Max_Helix_Propensity': max_helix_prop,
            'Avg_Helix_Propensity': avg_helix_prop,
            'Max_Transmembrane_Propensity': max_trans_prop,
            'Avg_Transmembrane_Propensity': avg_trans_prop,
        }
        
        features_data.append(feature_dict)
    
    return pd.DataFrame(features_data)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate the model performance using various metrics.
    
    Parameters:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {
        'Classification_Report': classification_report(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'F1_Score': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'Confusion_Matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics

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

def plot_amino_acid_composition(FN_composition: dict, bench_composition: dict, train_composition: dict):
    """
    Plot amino acid composition for False Negatives, Benchmark Positives, and Training Positives.
    
    Parameters:
        FN_composition (dict): Amino acid composition for False Negatives.
        bench_composition (dict): Amino acid composition for Benchmark Positives.
        train_composition (dict): Amino acid composition for Training Positives.
    """
    def composition_to_df(composition, label):
        return pd.DataFrame([
            {'Amino Acid': aa, 'Percentage': composition.get(aa, 0), 'Set': label}
            for aa in AMINO_ACIDS
        ])
    
    FN_comp_df = composition_to_df(FN_composition, 'False Negative')
    bench_comp_df = composition_to_df(bench_composition, 'Benchmark Positive')
    train_comp_df = composition_to_df(train_composition, 'Training Positive')
    
    combined_df = pd.concat([FN_comp_df, bench_comp_df, train_comp_df])
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=combined_df, x='Amino Acid', y='Percentage', hue='Set', palette='Set1')
    plt.xlabel('Amino Acid')
    plt.ylabel('Frequency')
    plt.title('Amino Acid Composition of Sequences up to 22 Residues')
    plt.grid(True)
    plt.show()

def plot_length_distribution(train_lengths: list, bench_tp_lengths: list, fn_lengths: list):
    """
    Plot the distribution of sequence lengths for training positives, benchmark true positives, and false negatives.
    
    Parameters:
        train_lengths (list): Sequence lengths for training positives.
        bench_tp_lengths (list): Sequence lengths for benchmark true positives.
        fn_lengths (list): Sequence lengths for false negatives.
    """
    bins = np.linspace(
        min(train_lengths + bench_tp_lengths + fn_lengths),
        max(train_lengths + bench_tp_lengths + fn_lengths),
        110
    )
    bar_width = (bins[1] - bins[0]) / 4
    
    plt.hist(train_lengths, bins=bins, density=True, alpha=0.5, label='Training Positive',
             color='blue', width=bar_width, align='mid', histtype='bar')
    plt.hist(bench_tp_lengths, bins=bins + bar_width, density=True, alpha=0.5, label='Benchmark True Positive',
             color='green', width=bar_width, align='mid', histtype='bar')
    plt.hist(fn_lengths, bins=bins + 2 * bar_width, density=True, alpha=0.5, label='False Negatives',
             color='red', width=bar_width, align='mid', histtype='bar')
    
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Relative Density')
    plt.xlim(0, 100)
    plt.legend(loc='best')
    plt.show()

def plot_boxplot_propensity(FN_features: pd.DataFrame, TP_features: pd.DataFrame, propensity_type: str):
    """
    Plot boxplots for propensities comparing False Negatives and True Positives.
    
    Parameters:
        FN_features (pd.DataFrame): Features for False Negatives.
        TP_features (pd.DataFrame): Features for True Positives.
        propensity_type (str): Type of propensity ('Helix' or 'Transmembrane').
    """
    if propensity_type.lower() == 'helix':
        max_prop = 'Max_Helix_Propensity'
        avg_prop = 'Avg_Helix_Propensity'
    elif propensity_type.lower() == 'transmembrane':
        max_prop = 'Max_Transmembrane_Propensity'
        avg_prop = 'Avg_Transmembrane_Propensity'
    else:
        raise ValueError("propensity_type must be either 'helix' or 'transmembrane'")
    
    fn_values = pd.DataFrame({
        'Propensity_Type': ['Max'] * len(FN_features) + ['Avg'] * len(FN_features),
        'Propensity_Value': pd.concat([FN_features[max_prop], FN_features[avg_prop]]),
        'Group': ['False Negative'] * (2 * len(FN_features))
    })
    
    tp_values = pd.DataFrame({
        'Propensity_Type': ['Max'] * len(TP_features) + ['Avg'] * len(TP_features),
        'Propensity_Value': pd.concat([TP_features[max_prop], TP_features[avg_prop]]),
        'Group': ['True Positive'] * (2 * len(TP_features))
    })
    
    combined_df = pd.concat([fn_values, tp_values])
    
    plt.figure(figsize=(6, 6))
    sns.boxplot(x='Propensity_Type', y='Propensity_Value', hue='Group', data=combined_df)
    plt.title(f'Boxplot of {propensity_type.capitalize()} Propensities')
    plt.show()

def main():
    # Paths to input files
    positive_fasta = "cluster-results_pos_rep_seq.fasta"
    negative_fasta = "cluster-results_neg_rep_seq.fasta"
    split_pos = "split_pos.tsv"
    split_neg = "split_neg.tsv"
    
    # Load data
    positive_df = get_dataframe(BASE_PATH, split_pos, positive_fasta, class_label=1)
    negative_df = get_dataframe(BASE_PATH, split_neg, negative_fasta, class_label=0)
    
    # Combine positive and negative dataframes
    positive_negative_df = pd.concat([positive_df, negative_df], ignore_index=True)
    
    # Split into benchmarking and training sets
    benchmarking_df = positive_negative_df[positive_negative_df["set"] == "B"].reset_index(drop=True)
    training_df = positive_negative_df[positive_negative_df["set"] == "T"].reset_index(drop=True)
    
    # Save the combined DataFrame
    training_df.to_csv(os.path.join(BASE_PATH, "positive_negative_df.csv"), index=False)
    print("\nAll results have been saved to 'positive_negative_df.csv'")
    
    n_folds = 5  # Number of cross-validation folds
    
    # Define the parameter grid for C and gamma
    param_grid = {'C': [1, 2, 4, 8], 'gamma': [0.5, 1, 2, 'scale']}
    
    # Define feature combinations
    feature_combinations = [
        # Baseline Only
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS],
        
        # Two Features
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS] + ['Max_KD', 'Avg_KD'],
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS] + ['Max_Charge', 'Normalized_Charge_Position'],
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS] + ['Max_Helix_Propensity', 'Avg_Helix_Propensity'],
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS] + ['Max_Transmembrane_Propensity', 'Avg_Transmembrane_Propensity'],
        
        # Three Features
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS] + ['Max_KD', 'Avg_KD', 'Max_Charge', 'Normalized_Charge_Position'],
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS] + ['Max_KD', 'Avg_KD', 'Max_Helix_Propensity', 'Avg_Helix_Propensity'],
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS] + ['Max_Transmembrane_Propensity', 'Avg_Transmembrane_Propensity'],
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS] + ['Max_KD', 'Avg_KD', 'Max_Helix_Propensity', 'Avg_Helix_Propensity'],
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS] + ['Max_Charge', 'Normalized_Charge_Position', 'Max_Helix_Propensity', 'Avg_Helix_Propensity'],
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS] + ['Max_Charge', 'Normalized_Charge_Position', 'Max_Transmembrane_Propensity', 'Avg_Transmembrane_Propensity'],
        
        # All Features
        [f'AA_Percentage_{aa}' for aa in AMINO_ACIDS] + [
            'Max_Charge', 'Normalized_Charge_Position',
            'Max_Helix_Propensity', 'Avg_Helix_Propensity',
            'Max_Transmembrane_Propensity', 'Avg_Transmembrane_Propensity'
        ]
    ]
    
    # Initialize a list to store results for each feature combination
    results_list = []
    
    # Iterate over each feature combination
    for idx, current_features in enumerate(feature_combinations, 1):
        print(f"\n=== Evaluating Feature Combination {idx}/{len(feature_combinations)} ===")
        print(f"Features Included: {current_features}\n")
        
        # Initialize a dictionary to store MCC and other metrics for each (C, gamma) combination
        combination_results = defaultdict(lambda: {'mcc': [], 'sen': [], 'ppv': [], 'acc': []})
        
        # Perform cross-validation for this feature combination
        for fold in range(n_folds):
            print(f"  Fold {fold + 1}/{n_folds}")
            
            # Split the data into test, validation, and training sets
            test_set, validation_set, training_set, _, _, _ = split_data(training_df, fold, n_folds=n_folds)
            
            # Extract features for each subset
            training_feature_df = extract_features(training_set)
            validation_feature_df = extract_features(validation_set)
            test_feature_df = extract_features(test_set)
            
            # Select only the current features
            X_train = training_feature_df[current_features]
            y_train = training_feature_df['Class']
            X_val = validation_feature_df[current_features]
            y_val = validation_feature_df['Class']
            
            # Initialize and fit the scaler
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Iterate over all combinations of C and gamma
            for c in param_grid['C']:
                for g in param_grid['gamma']:
                    # Initialize the SVM model with current parameters
                    current_svm = SVC(C=c, gamma=g, kernel='rbf')
                    
                    # Train the model on the scaled training set
                    current_svm.fit(X_train_scaled, y_train)
                    
                    # Predict on the scaled validation set
                    y_pred = current_svm.predict(X_val_scaled)
                    
                    # Calculate performance metrics
                    mcc = matthews_corrcoef(y_val, y_pred)
                    sen = recall_score(y_val, y_pred, zero_division=0)
                    ppv = precision_score(y_val, y_pred, zero_division=0)
                    acc = accuracy_score(y_val, y_pred)
                    
                    # Store the metrics
                    combination_results[(c, g)]['mcc'].append(mcc)
                    combination_results[(c, g)]['sen'].append(sen)
                    combination_results[(c, g)]['ppv'].append(ppv)
                    combination_results[(c, g)]['acc'].append(acc)
                    
                    print(f"    Evaluated C={c}, gamma={g} => MCC={mcc:.4f}, SEN={sen:.4f}, PPV={ppv:.4f}, ACC={acc:.4f}")
        
        # Compute average metrics for each (C, gamma) combination
        average_results = {
            params: {metric: np.mean(values) for metric, values in metrics.items()}
            for params, metrics in combination_results.items()
        }
        
        # Identify the (C, gamma) combination with the highest average MCC
        best_params = max(average_results, key=lambda x: average_results[x]['mcc'])
        best_c, best_g = best_params
        best_mcc = average_results[best_params]['mcc']
        best_sen = average_results[best_params]['sen']
        best_ppv = average_results[best_params]['ppv']
        best_acc = average_results[best_params]['acc']
        
        print(f"\n  Best Parameters for This Feature Combination:")
        print(f"  C={best_c}, gamma={best_g}")
        print(f"  Average MCC={best_mcc:.4f}, SEN={best_sen:.4f}, PPV={best_ppv:.4f}, ACC={best_acc:.4f}\n")
        
        # Append the results to the results list
        results_list.append({
            'Feature_Combination': current_features,
            'Best_C': best_c,
            'Best_Gamma': best_g,
            'Average_MCC': best_mcc,
            'Average_SEN': best_sen,
            'Average_PPV': best_ppv,
            'Average_ACC': best_acc
        })
    
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Identify the overall best feature combination based on Average MCC
    overall_best = results_df.loc[results_df['Average_MCC'].idxmax()]
    print("=== Overall Best Feature Combination ===")
    print(f"Features Included: {overall_best['Feature_Combination']}")
    print(f"Best C: {overall_best['Best_C']}")
    print(f"Best Gamma: {overall_best['Best_Gamma']}")
    print(f"Average MCC: {overall_best['Average_MCC']:.4f}")
    print(f"Average SEN: {overall_best['Average_SEN']:.4f}")
    print(f"Average PPV: {overall_best['Average_PPV']:.4f}")
    print(f"Average ACC: {overall_best['Average_ACC']:.4f}")
    
    # Save the results to a CSV file
    results_df.to_csv(os.path.join(BASE_PATH, "svm_feature_combinations_results.csv"), index=False)
    print("\nAll results have been saved to 'svm_feature_combinations_results.csv'")
    
    # === TRAIN THE FINAL MODEL ===
    print("\n=== Training the Final Model ===")
    
    # Extract features for the entire training dataset
    full_feature_df = extract_features(training_df)
    X_full = full_feature_df.drop(['ID', 'Class'], axis=1)
    y_full = full_feature_df['Class']
    
    # Select the best feature combination
    best_feature_combination = overall_best['Feature_Combination']
    X_full_selected = full_feature_df[best_feature_combination]
    
    # Initialize and fit the scaler
    final_scaler = MinMaxScaler()
    final_scaler.fit(X_full_selected)
    X_full_scaled = final_scaler.transform(X_full_selected)
    
    # Initialize and train the final SVM model
    final_svm = SVC(C=overall_best['Best_C'], gamma=overall_best['Best_Gamma'], kernel='rbf')
    final_svm.fit(X_full_scaled, y_full)
    
    # Predict on the training set (for demonstration; typically, use separate validation)
    y_pred_full = final_svm.predict(X_full_scaled)
    
    # === TEST ON BENCHMARKING ===
    print("\n=== Evaluating on Benchmark Set ===")
    
    # Extract features for the benchmarking set
    benchmarking_features = extract_features(benchmarking_df)
    X_benchmark = benchmarking_features[best_feature_combination]
    y_benchmark_true = benchmarking_features['Class']
    
    # Scale the benchmarking features
    X_benchmark_scaled = final_scaler.transform(X_benchmark)
    
    # Predict using the final model
    y_benchmark_pred = final_svm.predict(X_benchmark_scaled)
    
    # Evaluate the predictions
    benchmark_metrics = evaluate_model(y_benchmark_true.values, y_benchmark_pred)
    
    # Print evaluation metrics
    print("=== Classification Report ===")
    print(benchmark_metrics['Classification_Report'])
    print(f"Accuracy: {benchmark_metrics['Accuracy']:.2f}")
    print(f"Matthews Correlation Coefficient (MCC): {benchmark_metrics['MCC']:.2f}")
    print(f"F1 Score: {benchmark_metrics['F1_Score']:.2f}")
    print(f"Precision: {benchmark_metrics['Precision']:.2f}")
    print(f"Recall: {benchmark_metrics['Recall']:.2f}")
    print("=== Confusion Matrix ===")
    print(benchmark_metrics['Confusion_Matrix'])
    
    # === ANALYSIS === #
    print("\n=== Detailed Analysis ===")
    
    # Identify False Positives and False Negatives
    FP_indices = np.where((y_benchmark_pred == 1) & (y_benchmark_true == 0))[0]
    FN_indices = np.where((y_benchmark_pred == 0) & (y_benchmark_true == 1))[0]
    
    FP_list = benchmarking_df.loc[FP_indices, 'ID'].tolist()
    FN_list = benchmarking_df.loc[FN_indices, 'ID'].tolist()
    
    FP_df = benchmarking_df[benchmarking_df["ID"].isin(FP_list)]
    FN_df = benchmarking_df[benchmarking_df["ID"].isin(FN_list)]
    
    # Plot Kingdom Distribution
    plot_kingdom_distribution(benchmarking_df, 'Distribution of Kingdoms (Benchmark)')
    plot_kingdom_distribution(FP_df, 'Distribution of Kingdoms (False Positives)')
    plot_kingdom_distribution(FN_df, 'Distribution of Kingdoms (False Negatives)')
    
    # Compute False Positive Rate
    negative_bench_df = benchmarking_df[benchmarking_df["Class"] == 0]
    FPR = len(FP_list) / negative_bench_df.shape[0] if negative_bench_df.shape[0] > 0 else 0
    print(f"False Positive Rate: {FPR:.4f}")
    
    # Compute Transmembrane Domain Misclassification
    FP_TM = FP_df["Cleavage_pos"].sum()
    Neg_TM = negative_bench_df["Cleavage_pos"].sum()
    FPR_TM = FP_TM / Neg_TM if Neg_TM > 0 else 0
    print(f"FP_TM: {FP_TM}")
    print(f"Neg_TM: {Neg_TM}")
    print(f"The fraction of negatives having the transmembrane misclassified: {FPR_TM:.4f}")
    print(f"The percentage of False Positives having the transmembrane domain: {FP_TM / len(FP_list) if len(FP_list) > 0 else 0:.4f}")
    
    # Identify True Positives
    bench_pos_df = benchmarking_df[benchmarking_df["Class"] == 1].reset_index(drop=True)
    TP_df = bench_pos_df[~bench_pos_df['ID'].isin(FN_df['ID'])]
    
    print(f"True Positives: {TP_df.shape}")
    print(f"False Negatives: {FN_df.shape}")
    print(f"Total Benchmark Positives: {bench_pos_df.shape}")
    
    # Amino Acid Composition Comparison
    FN_seq = FN_df['First_40_AAs'].apply(lambda seq: seq[:22]).tolist()
    TP_seq = TP_df['First_40_AAs'].apply(lambda seq: seq[:22]).tolist()
    train_pos_seq = training_df[training_df["Class"] == 1]['First_40_AAs'].apply(lambda seq: seq[:22]).tolist()
    
    def calculate_aa_composition(sequence_list: list) -> dict:
        """Calculate the amino acid composition for a list of sequences."""
        concatenated_seq = ''.join(sequence_list)
        analysis = ProteinAnalysis(concatenated_seq)
        return analysis.get_amino_acids_percent()
    
    FN_composition = calculate_aa_composition(FN_seq)
    TP_composition = calculate_aa_composition(TP_seq)
    train_composition = calculate_aa_composition(train_pos_seq)
    
    plot_amino_acid_composition(FN_composition, TP_composition, train_composition)
    
    # Sequence Length Distribution
    seq_len_FN = FN_df['Cleavage_pos'].tolist()
    seq_len_TP = TP_df['Cleavage_pos'].tolist()
    seq_len_train_pos = training_df[training_df["Class"] == 1]['Cleavage_pos'].tolist()
    
    plot_length_distribution(seq_len_train_pos, seq_len_TP, seq_len_FN)
    
    # Propensity Boxplots
    TP_features = extract_features(TP_df)
    FN_features = extract_features(FN_df)
    
    plot_boxplot_propensity(FN_features, TP_features, 'helix')
    plot_boxplot_propensity(FN_features, TP_features, 'transmembrane')

if __name__ == "__main__":
    main()