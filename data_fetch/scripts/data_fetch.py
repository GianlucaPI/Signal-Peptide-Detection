import os
import re
import json
import random
import subprocess
from abc import ABC, abstractmethod
from collections import Counter
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from Bio import SeqIO
from Bio.Data import IUPACData
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from matplotlib import pyplot as plt
from requests.adapters import HTTPAdapter, Retry

# ----------------------------
# Data Fetching and Processing
# ----------------------------

class UniProtEntryProcessor(ABC):
    """
    Abstract base class for processing UniProt entries.
    Defines the interface for filtering and extracting fields from entries.
    """
    @abstractmethod
    def filter_entry(self, entry: dict) -> bool:
        """
        Determines whether a UniProt entry should be included in the dataset.

        :param entry: A dictionary representing a UniProt entry.
        :return: True if the entry passes the filter criteria, False otherwise.
        """
        pass

    @abstractmethod
    def extract_fields(self, entry: dict) -> tuple:
        """
        Extracts relevant fields from a UniProt entry.

        :param entry: A dictionary representing a UniProt entry.
        :return: A tuple containing extracted fields.
        """
        pass

class PositiveUniProtEntryProcessor(UniProtEntryProcessor):
    """
    Processor for positive UniProt entries.
    Filters entries based on Signal peptide features and extracts relevant fields.
    """
    def filter_entry(self, entry: dict) -> bool:
        """
        Filters entries to include only those with a Signal peptide feature
        where the end position is greater than 13 and not marked as 'not cleaved'.

        :param entry: A dictionary representing a UniProt entry.
        :return: True if the entry meets the criteria, False otherwise.
        """
        features = entry.get('features', [])
        for feature in features:
            if feature.get('type') == 'Signal':
                end = feature.get('location', {}).get('end', {}).get('value')
                description = feature.get('description', '').lower()

                if end is not None and int(end) > 13 and 'not cleaved' not in description:
                    return True
        return False

    def extract_fields(self, entry: dict) -> tuple:
        """
        Extracts primary accession, scientific name, lineage, sequence length,
        cleavage position, and sequence value from a positive entry.
        """
        # Initialize cleavage position
        cleavage_pos = None

        # Extract the cleavage position from the Signal feature
        for feature in entry.get("features", []):
            if feature.get("type") == "Signal":
                end = feature.get("location", {}).get("end", {}).get("value")
                description = feature.get("description", "").lower()
                if end is not None and int(end) > 13 and 'not cleaved' not in description:
                    cleavage_pos = int(end)
                    break  # Assuming only one relevant Signal feature

        # If no valid cleavage position found, set to a default value or handle accordingly
        if cleavage_pos is None:
            cleavage_pos = 0  # Or consider skipping this entry

        # Determine the lineage
        lineages = ["Metazoa", "Fungi", "Viridiplantae"]
        organism_lineage = entry.get("organism", {}).get("lineage", [])
        lineage = next((l for l in lineages if l in organism_lineage), "Others")

        # Correctly calculate sequence length
        sequence = entry.get("sequence", {}).get("value", "")
        seq_len = len(sequence)

        return (
            entry.get("primaryAccession", ""),
            entry.get("organism", {}).get("scientificName", ""),
            lineage,
            seq_len,  # Correct sequence length
            cleavage_pos,
            sequence
        )

class NegativeUniProtEntryProcessor(UniProtEntryProcessor):
    """
    Processor for negative UniProt entries.
    Filters entries based on Transmembrane features and extracts relevant fields.
    """

    def filter_entry(self, entry: dict) -> bool:
        """
        Filters entries to include those that have any Transmembrane features.

        :param entry: A dictionary representing a UniProt entry.
        :return: True if the entry meets the criteria, False otherwise.
        """
        features = entry.get('features', [])
        for feature in features:
            if feature.get('type') == 'Transmembrane':
                return True
        return False

    def extract_fields(self, entry: dict) -> tuple:
        """
        Extracts primary accession, scientific name, lineage, sequence length,
        transmembrane boolean, and sequence value from a negative entry.

        :param entry: A dictionary representing a UniProt entry.
        :return: A tuple containing the extracted fields.
        """
        # Extract kingdom information
        try:
            kingdom = next(
                l["scientificName"] for l in entry.get("lineages", [])
                if l.get("rank") == "kingdom"
            )
        except StopIteration:
            kingdom = "Other"

        # Determine if entry has a Transmembrane feature with specific criteria
        tm = False
        for feature in entry.get("features", []):
            if feature.get("type") == "Transmembrane":
                description = feature.get("description", "")
                if re.search("Helical", description):
                    start = feature.get("location", {}).get("start", {}).get("value", 0)
                    if start <= 90:
                        tm = True
                        break

        # Determine the lineage
        lineages = ["Metazoa", "Fungi", "Viridiplantae"]
        organism_lineage = entry.get("organism", {}).get("lineage", [])
        lineage = next((l for l in lineages if l in organism_lineage), "Others")

        # Correctly calculate sequence length
        sequence = entry.get("sequence", {}).get("value", "")
        seq_len = len(sequence)

        return (
            entry.get("primaryAccession", ""),
            entry.get("organism", {}).get("scientificName", ""),
            lineage,
            seq_len,  # Correct sequence length
            tm,
            sequence
        )

        return (
            entry.get("primaryAccession", ""),
            entry.get("organism", {}).get("scientificName", ""),
            lineage,
            entry.get("sequence", {}).get("length", 0),
            tm,
            entry.get("sequence", {}).get("value", "")#.replace("S","U") #replace "S" with "U"
        )

class UniProtDatasetFetcher:
    """
    Fetches datasets from UniProt using provided base URLs and entry processors.
    Handles HTTP requests with retries and pagination.
    """

    def __init__(self, base_url: str, entry_processor: UniProtEntryProcessor, batch_size: int = 500):
        """
        Initializes the dataset fetcher.

        :param base_url: The base URL for UniProt API queries.
        :param entry_processor: An instance of UniProtEntryProcessor to process entries.
        :param batch_size: Number of entries to fetch per request.
        """
        self.base_url = base_url
        self.batch_size = batch_size
        self.entry_processor = entry_processor
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """
        Creates a requests session with retry strategy.

        :return: A configured requests.Session object.
        """
        retries = Retry(
            total=5,
            backoff_factor=0.25,
            status_forcelist=[500, 502, 503, 504]
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def _get_next_link(self, headers: dict) -> str:
        """
        Parses the 'Link' header to find the next page URL.

        :param headers: HTTP response headers.
        :return: URL string for the next page or None if not found.
        """
        if "Link" in headers:
            re_next_link = re.compile(r'<(.+)>; rel="next"')
            match = re_next_link.search(headers["Link"])
            if match:
                return match.group(1)
        return None

    def _get_batch(self, batch_url: str):
        """
        Generator that yields batches of entries and the total count.

        :param batch_url: URL to fetch the current batch.
        """
        while batch_url:
            response = self.session.get(batch_url)
            response.raise_for_status()
            total = response.headers.get("x-total-results", "0")
            yield response, int(total)
            batch_url = self._get_next_link(response.headers)

    def fetch_dataset(self):
        """
        Fetches the dataset, yielding entries that pass the filter criteria.

        :yield: Filtered UniProt entries.
        """
        n_total, n_filtered = 0, 0
        url = f"{self.base_url}&size={self.batch_size}"

        for batch, total in self._get_batch(url):
            batch_json = batch.json()
            for entry in batch_json.get("results", []):
                n_total += 1
                if self.entry_processor.filter_entry(entry):
                    n_filtered += 1
                    yield entry

        print(f"Total entries fetched: {n_total}, Entries after filtering: {n_filtered}")

# ----------------------------
# File Handling
# ----------------------------

class UniProtFileCreator:
    """
    Creates TSV and FASTA files from processed UniProt entries.
    """

    def __init__(self, output_file_name: str):
        """
        Initializes the file creator.

        :param output_file_name: Base name for the output files.
        """
        self.output_file_name = output_file_name

    def create_tsv_file(self, data_generator, entry_processor: UniProtEntryProcessor):
        """
        Creates a TSV file from the filtered entries.

        :param data_generator: Generator yielding filtered UniProt entries.
        :param entry_processor: Processor to extract fields from entries.
        """
        tsv_file = f"{self.output_file_name}.tsv"
        fasta_file = f"{self.output_file_name}.fasta"

        with open(tsv_file, "w") as tsv_out, open(fasta_file, "w") as fasta_out:
            for entry in data_generator:
                fields = entry_processor.extract_fields(entry)
                # Write TSV (excluding the sequence value)
                tsv_out.write("\t".join(map(str, fields[:-1])) + "\n")
                # Write FASTA
                fasta_out.write(f">{fields[0]}\n{fields[-1]}\n")

        print(f"TSV file created at {tsv_file}")
        print(f"FASTA file created at {fasta_file}")

class RepresentativeFileParser:
    """
    Parses representative files to create DataFrames and split datasets into training and benchmarking sets.
    """

    def __init__(self, input_fasta: str, output_tsv: str, original_tsv: str, split_tsv: str):
        """
        Initializes the file parser.

        :param input_fasta: Path to the input FASTA file containing representative sequences.
        :param output_tsv: Path to the output TSV file for representative data.
        :param original_tsv: Path to the original TSV file containing all entries.
        :param split_tsv: Path to the split TSV file for training and benchmarking sets.
        """
        self.input_fasta = input_fasta
        self.output_tsv = output_tsv
        self.original_tsv = original_tsv
        self.split_tsv = split_tsv
        self.unique_elements = []
        self.df = pd.DataFrame()
        self.file_creator = None  # Placeholder for UniProtFileCreator

    def get_unique_element_information(self):
        """
        Extracts unique accession IDs from the input FASTA file.
        """
        with open(self.input_fasta, "r") as file_reader:
            for line in file_reader:
                if line.startswith(">"):
                    accession_id = line[1:].strip()
                    self.unique_elements.append(accession_id)
        print(f"Extracted {len(self.unique_elements)} unique accession IDs from {self.input_fasta}")

    def print_representative(self):
        """
        Writes representative entries to the output TSV file based on unique accession IDs.
        """
        with open(self.original_tsv, "r") as original_file, open(self.output_tsv, "w") as output_file:
            for line in original_file:
                fields = line.strip().split("\t")
                accession_id = fields[0]
                if accession_id in self.unique_elements:
                    output_file.write("\t".join(fields) + "\n")
        print(f"Representative TSV file created at {self.output_tsv}")

    def create_split_tsv(self) -> pd.DataFrame:
        """
        Creates a DataFrame from the representative TSV file, shuffles it, and splits into training and benchmarking sets.

        :return: A pandas DataFrame with the split information.
        """
        self.df = pd.read_csv(
            self.output_tsv, sep="\t",
            names=["ID", "Organisms", "Kingdom", "Seq_len", "Cleavage_pos"]
        )
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle rows

        # Split into 80% training and 20% benchmarking
        training_cutoff = int(len(self.df) * 0.8)
        self.df['set'] = ['T' if i < training_cutoff else 'B' for i in range(len(self.df))]

        # Assign folds to the training set
        num_classes = 5
        class_assignments = [i % num_classes for i in range(len(self.df[self.df['set'] == 'T']))]
        random.shuffle(class_assignments)
        self.df.loc[self.df['set'] == 'T', 'Fold'] = class_assignments
        self.df['Fold'] = self.df['Fold'].astype('Int64')

        # Save the split DataFrame to a TSV file
        self.df.to_csv(f"{self.split_tsv}.tsv", sep="\t", index=False)
        print(f"Data split into training and benchmarking sets saved at {self.split_tsv}.tsv")

        return self.df

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the created DataFrame.

        :return: A pandas DataFrame with split information.
        """
        return self.df

'''
class MergeSplitDatasets:

    
    #Merges the split_pos.tsv and split_neg.tsv in a merged_split_posneg.tsv with their 
    #first 50 residues taken from the positive.fasta and negative.fasta
    

    def __init__(self, base_path: str, split_pos: str, split_neg: str, positive_fasta: str, negative_fasta: str):
        self.base_path = base_path
        self.split_pos = split_pos
        self.split_neg = split_neg
        self.positive_fasta = positive_fasta
        self.negative_fasta = negative_fasta
    
    def addSequenceToTSV(self, ):
        df = pd.read_csv(f"{self.split_pos}", sep = "\t")
        print(df.head())
'''         

# ----------------------------
# Clustering
# ----------------------------

class ClusterManager:
    """
    Manages clustering of sequences using MMSeqs.
    """

    def __init__(self, base_path: str):
        """
        Initializes the ClusterManager.

        :param base_path: Base directory path where data files are located.
        """
        self.base_path = base_path

    def run_mmseqs_cluster(self, input_fasta: str, cluster_output: str, min_seq_id: float = 0.3,
                           coverage: float = 0.4, cluster_mode: int = 1):
        """
        Runs MMSeqs clustering on the provided FASTA file.

        :param input_fasta: Path to the input FASTA file.
        :param cluster_output: Base name for the clustering output files.
        :param min_seq_id: Minimum sequence identity for clustering.
        :param coverage: Coverage threshold for clustering.
        :param cluster_mode: Clustering mode parameter for MMSeqs.
        """
        command = [
            "mmseqs", "easy-cluster",
            input_fasta,
            cluster_output,
            "tmp",  # Temporary directory
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
            "--cov-mode", "0",
            "--cluster-mode", str(cluster_mode)
        ]

        print(f"Running MMSeqs clustering: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
            print(f"Clustering completed for {input_fasta}.")
        except subprocess.CalledProcessError as e:
            print(f"Error during MMSeqs clustering: {e}")

# ----------------------------
# Data Analysis
# ----------------------------

class SequenceLengthAnalyzer:
    """
    Analyzes the distribution of sequence lengths between positive and negative datasets.
    Generates various plots and prints statistics.
    """

    def __init__(self, positive_df: pd.DataFrame, negative_df: pd.DataFrame, set_type: str, column_name: str,
                 max_seq_length: int = 2000):
        """
        Initializes the SequenceLengthAnalyzer.

        :param positive_df: DataFrame containing positive sequences.
        :param negative_df: DataFrame containing negative sequences.
        :param set_type: Dataset type ('T' for Training, 'B' for Benchmarking).
        :param column_name: Column name containing sequence lengths.
        :param max_seq_length: Maximum sequence length to consider for analysis.
        """
        self.positive_df = positive_df
        self.negative_df = negative_df
        self.set_type = set_type
        self.column_name = column_name
        self.max_seq_length = max_seq_length
        self.positive_lengths = None
        self.negative_lengths = None
        self.plot_df_filtered = None
        self.plot_df_melted = None

    def filter_data(self):
        """
        Filters the sequence lengths based on the dataset type.
        """
        self.positive_lengths = self.positive_df[self.positive_df["set"] == self.set_type][self.column_name]
        self.negative_lengths = self.negative_df[self.negative_df["set"] == self.set_type][self.column_name]
        print(f"Filtered data for set '{self.set_type}'.")

    @staticmethod
    def calc_rel_freq(data: pd.Series, bins: np.ndarray) -> tuple:
        """
        Calculates relative frequencies for given data and bins.

        :param data: Series containing numerical data.
        :param bins: Array of bin edges.
        :return: Tuple of relative frequencies and bin edges.
        """
        hist, bin_edges = np.histogram(data, bins=bins)
        rel_freq = hist / len(data)
        return rel_freq, bin_edges

    def prepare_plot_data(self):
        """
        Prepares the data required for plotting by calculating relative frequencies.
        """
        max_length = max(self.positive_lengths.max(), self.negative_lengths.max())
        bins = np.arange(0, max_length + 50, 50)
        pos_freq, pos_edges = self.calc_rel_freq(self.positive_lengths, bins)
        neg_freq, neg_edges = self.calc_rel_freq(self.negative_lengths, bins)

        plot_df = pd.DataFrame({
            'bin_start': pos_edges[:-1],
            'Positive': pos_freq,
            'Negative': neg_freq
        })

        self.plot_df_filtered = plot_df[plot_df['bin_start'] <= self.max_seq_length]
        self.plot_df_melted = pd.melt(
            self.plot_df_filtered,
            id_vars=['bin_start'],
            var_name='Type',
            value_name='Frequency'
        )
        print("Prepared plot data.")

    def create_bar_plot(self):
        """
        Creates a bar plot comparing the relative frequencies of sequence lengths.
        """
        plt.figure(figsize=(15, 6))
        sns.barplot(
            data=self.plot_df_melted,
            x='bin_start',
            y='Frequency',
            hue='Type',
            palette=['blue','red']
        )
        plt.title(f'Relative Frequency of Sequence Lengths (Up to {self.max_seq_length})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Relative Frequency')
        plt.xticks(rotation=45)

        total_positive = len(self.positive_lengths)
        total_negative = len(self.negative_lengths)
        plt.legend(
            title='Type (Total Count)',
            labels=[f'Positive ({total_positive})', f'Negative ({total_negative})']
        )

        plt.tight_layout()
        plt.show()
        print("Bar plot generated.")

    def create_density_plot(self):
        """
        Creates a density plot comparing the distributions of sequence lengths.
        """
        plt.figure(figsize=(15, 6))
        sns.kdeplot(
            data=self.positive_lengths[self.positive_lengths <= self.max_seq_length],
            color='blue',
            label='Positive',
            linewidth=2
        )
        sns.kdeplot(
            data=self.negative_lengths[self.negative_lengths <= self.max_seq_length],
            color='red',
            label='Negative',
            linewidth=2
        )
        plt.title(f'Distribution of Sequence Lengths (Up to {self.max_seq_length})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Density')
        plt.xlim(0, self.max_seq_length)

        total_positive = len(self.positive_lengths)
        total_negative = len(self.negative_lengths)
        plt.legend(
            title='Type (Total Count)',
            labels=[f'Positive ({total_positive})', f'Negative ({total_negative})']
        )

        plt.tight_layout()
        plt.show()
        print("Density plot generated.")

    def create_box_plot(self):
        """
        Creates a box plot comparing the distribution of sequence lengths,
        excluding outliers.
        """
        plt.figure(figsize=(15, 6))

        def remove_outliers(data: pd.Series) -> pd.Series:
            """
            Removes outliers from the data using the IQR method.

            :param data: Series containing numerical data.
            :return: Series with outliers removed.
            """
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data >= lower_bound) & (data <= upper_bound)]

        # Remove outliers and filter by max_seq_length
        positive_data = remove_outliers(
            self.positive_lengths[self.positive_lengths <= self.max_seq_length]
        )
        negative_data = remove_outliers(
            self.negative_lengths[self.negative_lengths <= self.max_seq_length]
        )
        data_to_plot = [positive_data, negative_data]

        # Create box plot
        box_plot = plt.boxplot(
            data_to_plot,
            patch_artist=True,
            labels=['Positive', 'Negative']
        )

        # Customize colors
        colors = ['blue', 'red']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        plt.title(f'Distribution of Sequence Lengths (Outliers Removed)')
        plt.xlabel('Sequence Type')
        plt.ylabel('Sequence Length')

        # Set y-axis limit based on data without outliers
        plt.ylim(0, max(positive_data.max(), negative_data.max()) * 1.1)

        # Add count information to the x-axis labels
        total_positive = len(self.positive_lengths)
        total_negative = len(self.negative_lengths)
        plt.xticks(
            range(1, 3),
            [f'Positive\n(n={total_positive})', f'Negative\n(n={total_negative})']
        )

        plt.tight_layout()
        plt.show()
        print("Box plot generated.")

    def print_statistics(self):
        """
        Prints statistics about the sequence length distributions.
        """
        total_positive = len(self.positive_lengths)
        total_negative = len(self.negative_lengths)
        positive_percentage = (self.positive_lengths <= self.max_seq_length).mean() * 100
        negative_percentage = (self.negative_lengths <= self.max_seq_length).mean() * 100

        print(f"Total positive sequences: {total_positive}")
        print(f"Total negative sequences: {total_negative}")
        print("Percentage of data visualized in plots:")
        print(f"Positive: {positive_percentage:.2f}%")
        print(f"Negative: {negative_percentage:.2f}%")

    def analyze(self):
        """
        Performs the complete analysis by filtering data, preparing plot data,
        generating plots, and printing statistics.
        """
        self.filter_data()
        self.prepare_plot_data()
        self.create_bar_plot()
        self.create_density_plot()
        self.create_box_plot()
        self.print_statistics()

class SignalPeptideLengthAnalyzer:
    """
    Analyzes the distribution of Signal Peptide (SP) lengths in positive datasets.
    Generates various plots and prints statistics.
    """

    def __init__(self, positive_df: pd.DataFrame, set_type: str, column_name: str,
                 max_seq_length: int = 2000):
        """
        Initializes the SignalPeptideLengthAnalyzer.

        :param positive_df: DataFrame containing positive sequences.
        :param set_type: Dataset type ('T' for Training, 'B' for Benchmarking).
        :param column_name: Column name containing SP lengths.
        :param max_seq_length: Maximum SP length to consider for analysis.
        """
        self.positive_df = positive_df
        self.set_type = set_type
        self.column_name = column_name
        self.max_seq_length = max_seq_length
        self.positive_lengths = None
        self.positive_df_filtered = None

    def filter_data(self):
        """
        Filters the SP lengths based on the dataset type.
        """
        self.positive_lengths = self.positive_df[self.positive_df["set"] == self.set_type][self.column_name]
        self.positive_df_filtered = pd.DataFrame({'SP_Length': self.positive_lengths})
        print(f"Filtered SP lengths for set '{self.set_type}'.")

    def create_bar_plot(self):
        """
        Creates a histogram plot of SP lengths.
        """
        plt.figure(figsize=(15, 6))
        sns.histplot(
            data=self.positive_df_filtered,
            bins=250,
            x='SP_Length',
            color='skyblue'
        )
        plt.title(f'Distribution of Signal Peptide Lengths (Up to {self.max_seq_length})')
        plt.xlabel('Signal Peptide Length')
        plt.ylabel('Relative Frequency')

        total_positive = len(self.positive_lengths)
        plt.legend(
            title='Type (Total Count)',
            labels=[f'Positive ({total_positive})']
        )

        plt.tight_layout()
        plt.show()
        print("Histogram plot generated.")

    def create_density_plot(self):
        """
        Creates a density plot of SP lengths.
        """
        plt.figure(figsize=(15, 6))
        sns.kdeplot(
            data=self.positive_lengths[self.positive_lengths <= self.max_seq_length],
            color='blue',
            label='Positive',
            linewidth=2
        )
        plt.title(f'Distribution of Signal Peptide Lengths (Up to {self.max_seq_length})')
        plt.xlabel('Signal Peptide Length')
        plt.ylabel('Density')
        plt.xlim(0, self.max_seq_length)

        total_positive = len(self.positive_lengths)
        plt.legend(
            title='Type (Total Count)',
            labels=[f'Positive ({total_positive})']
        )

        plt.tight_layout()
        plt.show()
        print("Density plot generated.")

    def create_box_plot(self):
        """
        Creates a box plot of SP lengths, excluding outliers.
        """
        plt.figure(figsize=(15, 6))

        def remove_outliers(data: pd.Series) -> pd.Series:
            """
            Removes outliers from the data using the IQR method.

            :param data: Series containing numerical data.
            :return: Series with outliers removed.
            """
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data >= lower_bound) & (data <= upper_bound)]

        # Remove outliers and filter by max_seq_length
        positive_data = remove_outliers(
            self.positive_df_filtered['SP_Length'][self.positive_df_filtered['SP_Length'] <= self.max_seq_length]
        )
        data_to_plot = [positive_data]

        # Create box plot
        box_plot = plt.boxplot(
            data_to_plot,
            patch_artist=True,
            labels=['Positive']
        )

        # Customize colors
        colors = ['blue']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        plt.title('Distribution of Signal Peptide Lengths (Outliers Removed)')
        plt.xlabel('Sequence Type')
        plt.ylabel('Signal Peptide Length')

        # Set y-axis limit based on data without outliers
        plt.ylim(0, positive_data.max() * 1.1)

        # Add count information to the x-axis labels
        total_positive = len(self.positive_df_filtered)
        plt.xticks([1], [f'Positive\n(n={total_positive})'])

        plt.tight_layout()
        plt.show()
        print("Box plot generated.")

        # Print additional statistics
        print("Statistics of Signal Peptide Length Distribution (Outliers Removed):")
        print(f"Positive:")
        print(f"  Median: {np.median(positive_data):.2f}")
        print(f"  Mean: {np.mean(positive_data):.2f}")
        print(f"  Standard Deviation: {np.std(positive_data):.2f}")
        print(f"  Minimum: {np.min(positive_data):.2f}")
        print(f"  Maximum: {np.max(positive_data):.2f}")
        print(f"  Number of sequences after outlier removal: {len(positive_data)}")
        print(f"  Percentage of data retained: {(len(positive_data) / len(self.positive_df_filtered)) * 100:.2f}%")

    def print_statistics(self):
        """
        Prints statistics about the SP length distribution.
        """
        total_positive = len(self.positive_lengths)
        positive_percentage = (self.positive_lengths <= self.max_seq_length).mean() * 100

        print(f"Total positive SP sequences: {total_positive}")
        print("Percentage of data visualized in plots:")
        print(f"Positive: {positive_percentage:.2f}%")

    def analyze(self):
        """
        Performs the complete SP length analysis by filtering data, generating plots,
        and printing statistics.
        """
        self.filter_data()
        self.create_bar_plot()
        self.create_density_plot()
        self.create_box_plot()
        self.print_statistics()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from Bio import SeqIO

class AminoAcidFrequencyAnalyzer:
    """
    Analyzes the frequency of amino acids in the training set.
    Generates a DataFrame of frequencies and calculates average frequencies.
    """

    def __init__(self, split_tsv: str, fasta_file: str, amino_acids: list, set_type: str = 'T', max_seq_length: int = 2000):
        self.split_tsv = split_tsv
        self.fasta_file = fasta_file
        self.amino_acids = amino_acids
        self.set_type = set_type
        self.max_seq_length = max_seq_length
        self.df = None
        self.filtered_ids = None
        self.id_to_cleavage_pos = {}
        self.filtered_sequences = []
        self.seq_ids = []
        self.df_counts = None
        self.df_frequencies = None
        self.average_frequencies = None
        self.swissprot_frequencies = {
            "A":8.25, "C":1.38, "D":5.46, "E":6.71, "F":3.86,
            "G":7.07, "H":2.27, "I":5.91, "K":5.80, "L":9.64,
            "M":2.41, "N":4.06, "P":4.74, "Q":3.93, "R":5.52,
            "S":6.65, "T":5.36, "V":6.85, "W":1.10, "Y":2.92
        }

    @staticmethod
    def calc_rel_freq(data: pd.Series, bins: np.ndarray) -> tuple:
        """
        Calculates relative frequencies for given data and bins.

        :param data: Series containing numerical data.
        :param bins: Array of bin edges.
        :return: Tuple of relative frequencies and bin edges.
        """
        hist, bin_edges = np.histogram(data, bins=bins)
        rel_freq = hist / len(data)
        return rel_freq, bin_edges

    def load_and_filter_data(self):
        """
        Loads the split TSV file and filters for the specified set type.
        Creates a dictionary mapping IDs to cleavage positions.
        """
        # Load the split TSV file
        self.df = pd.read_csv(self.split_tsv, sep='\t')
        # Filter for the specified set type
        self.filtered_ids = self.df[self.df['set'] == self.set_type][['ID', 'Cleavage_pos']]
        # Create a dictionary for fast lookup
        self.id_to_cleavage_pos = dict(zip(self.filtered_ids['ID'], self.filtered_ids['Cleavage_pos']))
        print(f"Loaded {len(self.filtered_ids)} records for set '{self.set_type}' from {self.split_tsv}")

    def process_fasta(self):
        """
        Parses the FASTA file and slices sequences based on cleavage positions.
        Stores the sliced sequences and their corresponding IDs.
        """
        # Parse and filter sequences
        for seq_record in SeqIO.parse(self.fasta_file, 'fasta'):
            if seq_record.id in self.id_to_cleavage_pos:
                self.seq_ids.append(seq_record.id)
                cleavage_pos = self.id_to_cleavage_pos[seq_record.id]
                # Slice the sequence from 0 to cleavage_pos (inclusive)
                sliced_sequence = seq_record.seq[0:(cleavage_pos + 1)]  # Include the cleavage residue
                self.filtered_sequences.append(sliced_sequence)
        print(f"Processed {len(self.filtered_sequences)} sequences from {self.fasta_file}")

    def calculate_frequencies(self):
        """
        Counts amino acids in each sequence and calculates their frequencies.
        Creates a DataFrame of counts and frequencies.
        """
        data = []
        for seq in self.filtered_sequences:
            counts = Counter(seq)
            seq_length = len(seq)
            # Ensure all amino acids are included, even if count is zero
            seq_counts = {aa: counts.get(aa, 0) for aa in self.amino_acids}
            seq_counts['Sequence_Length'] = seq_length
            data.append(seq_counts)

        # Create DataFrame with counts
        self.df_counts = pd.DataFrame(data, index=self.seq_ids)
        print("Amino acid counts per sequence calculated.")

        # Calculate frequencies (percentage)
        self.df_frequencies = self.df_counts[self.amino_acids].div(self.df_counts['Sequence_Length'], axis=0) * 100
        print("Amino acid frequencies per sequence calculated.")
        print(self.df_counts['Sequence_Length'])

    def calculate_average_frequencies(self):
        """
        Calculates the average frequency of each amino acid across all sequences.
        """
        self.average_frequencies = self.df_frequencies.mean()
        print("Average amino acid frequencies calculated.")

    def generate_frequencies_dataframe(self):
        """
        Combines all steps to generate the frequencies DataFrame and calculate averages.
        """
        self.load_and_filter_data()
        self.process_fasta()
        self.calculate_frequencies()
        self.calculate_average_frequencies()

    def get_frequencies(self) -> pd.DataFrame:
        """
        Returns the DataFrame containing frequencies of amino acids.

        :return: DataFrame with amino acid frequencies.
        """
        return self.df_frequencies

    def get_average_frequencies(self) -> pd.Series:
        """
        Returns the Series containing average frequencies of amino acids.

        :return: Series with average amino acid frequencies.
        """
        return self.average_frequencies

    def prepare_plot_data(self):
        """
        Prepares the data required for plotting by calculating relative frequencies.
        """
        max_length = max(self.average_frequencies.max(), max(self.swissprot_frequencies.values()))
        bins = np.arange(0, max_length + 50, 50)
        pos_freq, pos_edges = self.calc_rel_freq(self.average_frequencies, bins)
        neg_freq, neg_edges = self.calc_rel_freq(np.array(list(self.swissprot_frequencies.values())), bins)

        plot_df = pd.DataFrame({
            'bin_start': pos_edges[:-1],
            'Positive': pos_freq,
            'Negative': neg_freq
        })

        self.plot_df_filtered = plot_df[plot_df['bin_start'] <= self.max_seq_length]
        
        self.plot_df_melted = pd.melt(
            self.plot_df_filtered,
            id_vars=['bin_start'],
            var_name='Type',
            value_name='Frequency'
        )
        print("Prepared plot data.")

    def create_bar_plot(self):
        """
        Creates a bar plot comparing the relative frequencies of sequence lengths.
        """
        plt.figure(figsize=(15, 6))
        sns.barplot(
            data=self.plot_df_melted,
            x='bin_start',
            y='Frequency',
            hue='Type',
            palette=['blue', 'red']
        )
        plt.title(f'Relative Frequency of Sequence Lengths (Up to {self.max_seq_length})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Relative Frequency')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
        print("Bar plot generated.")

    def create_density_plot(self):
        """
        Creates a density plot comparing the distributions of sequence lengths.
        """
        plt.figure(figsize=(15, 6))
        sns.kdeplot(
            data=self.df_frequencies.stack(),
            color='blue',
            label='Calculated',
            linewidth=2
        )
        sns.kdeplot(
            data=list(self.swissprot_frequencies.values()),
            color='red',
            label='SwissProt',
            linewidth=2
        )
        plt.title(f'Distribution of Amino Acid Frequencies')
        plt.xlabel('Frequency (%)')
        plt.ylabel('Density')
        plt.xlim(0, max(self.average_frequencies.max(), max(self.swissprot_frequencies.values())) * 1.1)

        plt.legend(
            title='Type',
            labels=['Calculated', 'SwissProt']
        )

        plt.tight_layout()
        plt.show()
        print("Density plot generated.")

    def create_box_plot(self):
        """
        Creates a box plot comparing the distribution of amino acid frequencies,
        excluding outliers.
        """
        plt.figure(figsize=(15, 6))

        def remove_outliers(data: pd.Series) -> pd.Series:
            """
            Removes outliers from the data using the IQR method.

            :param data: Series containing numerical data.
            :return: Series with outliers removed.
            """
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data >= lower_bound) & (data <= upper_bound)]

        # Remove outliers and filter by max_seq_length
        # Since we're dealing with frequencies, max_seq_length isn't directly relevant here
        calculated_freq = remove_outliers(self.average_frequencies)
        swissprot_freq = remove_outliers(pd.Series(self.swissprot_frequencies))

        data_to_plot = [calculated_freq, swissprot_freq]

        # Create box plot
        box_plot = plt.boxplot(
            data_to_plot,
            patch_artist=True,
            labels=['Calculated', 'SwissProt']
        )

        # Customize colors
        colors = ['blue', 'red']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        plt.title(f'Distribution of Amino Acid Frequencies (Outliers Removed)')
        plt.xlabel('Type')
        plt.ylabel('Frequency (%)')

        # Set y-axis limit based on data without outliers
        plt.ylim(0, max(calculated_freq.max(), swissprot_freq.max()) * 1.1)

        # Add count information to the x-axis labels
        total_calculated = len(calculated_freq)
        total_swissprot = len(swissprot_freq)
        plt.xticks(
            range(1, 3),
            [f'Calculated\n(n={total_calculated})', f'SwissProt\n(n={total_swissprot})']
        )

        plt.tight_layout()
        plt.show()
        print("Box plot generated.")

    def print_statistics(self):
        """
        Prints statistics about the sequence length distributions.
        """
        total_sequences = len(self.filtered_sequences)
        total_amino_acids = self.df_counts['Sequence_Length'].sum()
        calculated_percentage = 100  # Since all calculated frequencies are considered
        swissprot_percentage = 100  # Assuming full SwissProt data is used

        print(f"Total sequences analyzed: {total_sequences}")
        print(f"Total amino acids analyzed: {total_amino_acids}")
        print("Percentage of data visualized in plots:")
        print(f"Calculated: {calculated_percentage:.2f}%")
        print(f"SwissProt: {swissprot_percentage:.2f}%")

    def analyze(self):
        """
        Performs the complete analysis by generating frequencies and creating a side-by-side bar plot.
        """
        self.generate_frequencies_dataframe()
        
        # Create a DataFrame for plotting
        plot_data = pd.DataFrame({
            'Amino Acid': self.amino_acids,
            'Calculated Frequency': self.average_frequencies,
            'SwissProt Frequency': [self.swissprot_frequencies[aa] for aa in self.amino_acids]
        })

        # Melt the DataFrame for easier plotting
        plot_data_melted = pd.melt(plot_data, id_vars=['Amino Acid'], 
                                   var_name='Source', value_name='Frequency')

        # Create the plot
        plt.figure(figsize=(15, 8))
        sns.barplot(x='Amino Acid', y='Frequency', hue='Source', data=plot_data_melted)
        plt.title('Amino Acid Frequencies: Calculated vs SwissProt')
        plt.xlabel('Amino Acid')
        plt.ylabel('Frequency (%)')
        plt.xticks(rotation=45)
        plt.legend(title='Source')
        plt.tight_layout()
        plt.show()

        print("Side-by-side bar plot of amino acid frequencies generated.")
    
        # Print some statistics
        print("\nStatistics:")
        print(f"Total sequences analyzed: {len(self.filtered_sequences)}")
        print(f"Total amino acids analyzed: {self.df_counts['Sequence_Length'].sum()}")
        print("\nTop 5 most frequent amino acids in our data:")
        print(self.average_frequencies.sort_values(ascending=False).head())
        
        # Optional: Prepare additional plots
        self.prepare_plot_data()
        self.create_bar_plot()
        self.create_density_plot()
        self.create_box_plot()
        self.print_statistics()

class SwissProtFrequencyAnalyzer:
    """
    Analyzes the distribution of SwissProtFrequency and Average Residues.
    """

    def __init__(self, average_frequencies: pd.DataFrame, swissprot_frequencies: pd.DataFrame, set_type: str, column_name: str,
                 max_seq_length: int = 2000):
        """
        Initializes the SequenceLengthAnalyzer.

        :param positive_df: DataFrame containing positive sequences.
        :param negative_df: DataFrame containing negative sequences.
        :param set_type: Dataset type ('T' for Training, 'B' for Benchmarking).
        :param column_name: Column name containing sequence lengths.
        :param max_seq_length: Maximum sequence length to consider for analysis.
        """
        self.average_frequencies = average_frequencies
        self.swissprot_frequencies = swissprot_frequencies
        self.set_type = set_type
        self.column_name = column_name
        self.max_seq_length = max_seq_length
        self.positive_lengths = None
        self.negative_lengths = None
        self.plot_df_filtered = None
        self.plot_df_melted = None

    def filter_data(self):
        """
        Filters the sequence lengths based on the dataset type.
        """
        self.positive_lengths = self.positive_df[self.positive_df["set"] == self.set_type][self.column_name]
        self.negative_lengths = self.negative_df[self.negative_df["set"] == self.set_type][self.column_name]
        print(f"Filtered data for set '{self.set_type}'.")

    @staticmethod
    def calc_rel_freq(data: pd.Series, bins: np.ndarray) -> tuple:
        """
        Calculates relative frequencies for given data and bins.

        :param data: Series containing numerical data.
        :param bins: Array of bin edges.
        :return: Tuple of relative frequencies and bin edges.
        """
        hist, bin_edges = np.histogram(data, bins=bins)
        rel_freq = hist / len(data)
        return rel_freq, bin_edges

    def prepare_plot_data(self):
        """
        Prepares the data required for plotting by calculating relative frequencies.
        """
        max_length = max(self.average_frequencies.max(), self.swissprot_frequencies.max())
        bins = np.arange(0, max_length + 50, 50)
        pos_freq, pos_edges = self.calc_rel_freq(self.positive_lengths, bins)
        neg_freq, neg_edges = self.calc_rel_freq(self.negative_lengths, bins)

        plot_df = pd.DataFrame({
            'bin_start': pos_edges[:-1],
            'Positive': pos_freq,
            'Negative': neg_freq
        })

        self.plot_df_filtered = plot_df[plot_df['bin_start'] <= self.max_seq_length]
        self.plot_df_melted = pd.melt(
            self.plot_df_filtered,
            id_vars=['bin_start'],
            var_name='Type',
            value_name='Frequency'
        )
        print("Prepared plot data.")

    def create_bar_plot(self):
        """
        Creates a bar plot comparing the relative frequencies of sequence lengths.
        """
        plt.figure(figsize=(15, 6))
        sns.barplot(
            data=self.plot_df_melted,
            x='bin_start',
            y='Frequency',
            hue='Type',
            palette=['blue', 'red']
        )
        plt.title(f'Relative Frequency of Sequence Lengths (Up to {self.max_seq_length})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Relative Frequency')
        plt.xticks(rotation=45)

        total_frequencies = len(self.average_frequencies)
        swissprot_frequencies = len(self.swissprot_frequencies)
        plt.legend(
            title='Type (Total Count)',
            labels=[f'Average Fraquencies: ({total_frequencies})', f'Negative ({swissprot_frequencies})']
        )

        plt.tight_layout()
        plt.show()
        print("Bar plot generated.")

    def create_density_plot(self):
        """
        Creates a density plot comparing the distributions of sequence lengths.
        """
        plt.figure(figsize=(15, 6))
        sns.kdeplot(
            data=self.positive_lengths[self.positive_lengths <= self.max_seq_length],
            color='blue',
            label='Positive',
            linewidth=2
        )
        sns.kdeplot(
            data=self.negative_lengths[self.negative_lengths <= self.max_seq_length],
            color='red',
            label='Negative',
            linewidth=2
        )
        plt.title(f'Distribution of Sequence Lengths (Up to {self.max_seq_length})')
        plt.xlabel('Sequence Length')
        plt.ylabel('Density')
        plt.xlim(0, self.max_seq_length)

        total_positive = len(self.positive_lengths)
        total_negative = len(self.negative_lengths)
        plt.legend(
            title='Type (Total Count)',
            labels=[f'Positive ({total_positive})', f'Negative ({total_negative})']
        )

        plt.tight_layout()
        plt.show()
        print("Density plot generated.")

    def create_box_plot(self):
        """
        Creates a box plot comparing the distribution of sequence lengths,
        excluding outliers.
        """
        plt.figure(figsize=(15, 6))

        def remove_outliers(data: pd.Series) -> pd.Series:
            """
            Removes outliers from the data using the IQR method.

            :param data: Series containing numerical data.
            :return: Series with outliers removed.
            """
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data >= lower_bound) & (data <= upper_bound)]

        # Remove outliers and filter by max_seq_length
        positive_data = remove_outliers(
            self.positive_lengths[self.positive_lengths <= self.max_seq_length]
        )
        negative_data = remove_outliers(
            self.negative_lengths[self.negative_lengths <= self.max_seq_length]
        )
        data_to_plot = [positive_data, negative_data]

        # Create box plot
        box_plot = plt.boxplot(
            data_to_plot,
            patch_artist=True,
            labels=['Positive', 'Negative']
        )

        # Customize colors
        colors = ['blue', 'red']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        plt.title(f'Distribution of Sequence Lengths (Outliers Removed)')
        plt.xlabel('Sequence Type')
        plt.ylabel('Sequence Length')

        # Set y-axis limit based on data without outliers
        plt.ylim(0, max(positive_data.max(), negative_data.max()) * 1.1)

        # Add count information to the x-axis labels
        total_positive = len(self.positive_lengths)
        total_negative = len(self.negative_lengths)
        plt.xticks(
            range(1, 3),
            [f'Positive\n(n={total_positive})', f'Negative\n(n={total_negative})']
        )

        plt.tight_layout()
        plt.show()
        print("Box plot generated.")

    def analyze(self):
        """
        Performs the complete analysis by filtering data, preparing plot data,
        generating plots, and printing statistics.
        """
        self.filter_data()
        self.prepare_plot_data()
        self.create_bar_plot()
        self.create_density_plot()
        self.create_box_plot()
        self.print_statistics()

class KingdomDistributionAnalyzer:
    def __init__(self, positive_tsv_file_path, negative_tsv_file_path):
        self.positive_tsv_file_path = positive_tsv_file_path
        self.negative_tsv_file_path = negative_tsv_file_path
        self.df = None
        self.kingdom_counts = None

    def load_data(self):
        """Load the TSV file into a pandas DataFrame."""
        positive_df = pd.read_csv(self.positive_tsv_file_path, sep='\t')
        negative_df = pd.read_csv(self.negative_tsv_file_path, sep='\t')

        positive_negative_df = pd.concat([positive_df, negative_df], ignore_index=True)  # both positives and negatives

        self.df = positive_negative_df         #MERGE DATAFRAMES
        print(f"Data loaded. Shape: {self.df.shape}")

    def analyze_kingdom_distribution(self):
        """Analyze the distribution of kingdoms."""
        self.kingdom_counts = self.df['Kingdom'].value_counts()
        print("Kingdom distribution:")
        print(self.kingdom_counts)

    def create_pie_chart(self, title="Distribution of Kingdoms", save_path=None):
        """Create a pie chart of the kingdom distribution."""
        if self.kingdom_counts is None:
            print("Please run analyze_kingdom_distribution() first.")
            return

        plt.figure(figsize=(10, 8))
        plt.pie(self.kingdom_counts.values, labels=self.kingdom_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pie chart saved to {save_path}")
        
        plt.show()

    def run_analysis(self, save_path=None):
        """Run the complete analysis pipeline."""
        self.load_data()
        self.analyze_kingdom_distribution()
        self.create_pie_chart(save_path=save_path)

class MakeSequenceLogo:
    def __init__(self, base_path_url, positive_tsv_file_path, fasta_file_path):
        self.base_path_url = base_path_url
        self.positive_tsv_file_path = base_path_url + positive_tsv_file_path
        self.fasta_file_path = base_path_url + fasta_file_path
        self.df = None
        self.sequences = {}
        self.aligned_sequences = []
    
    def load_data(self):
        """Load the TSV files into a pandas DataFrame."""
        positive_df = pd.read_csv(self.positive_tsv_file_path, sep='\t')

        self.df = positive_df
        print(f"Data loaded. Shape: {self.df.shape}")
    
    def load_fasta_sequences(self):
        """Parse the FASTA file and map sequences to their corresponding IDs."""
        try:
            for record in SeqIO.parse(self.fasta_file_path, "fasta"):
                self.sequences[record.id] = str(record.seq)
            print(f"FASTA sequences loaded. Total sequences: {len(self.sequences)}")
        except FileNotFoundError:
            print(f"FASTA file not found at path: {self.fasta_file_path}")
    
    def extract_aligned_windows(self, window_size=15, output_file='sequence_for_logo.fasta'):
    
        # Initialize the list to store aligned sequences
        self.aligned_sequences = []
        
        # Open the output file in write mode
        with open(output_file, 'w') as fasta_file:
            # Iterate over each row in the dataframe
            for idx, row in self.df.iterrows():
                seq_id = row['ID']
                cleavage_pos = row['Cleavage_pos']
                sequence = self.sequences.get(seq_id, "")
                
                if not sequence:
                    print(f"Sequence for ID {seq_id} not found.")
                    continue
                
                # Adjust cleavage position to 0-based index
                cleavage_index = cleavage_pos - 1  # Assuming cleavage_pos is 1-based
                
                # Define window boundaries
                start = cleavage_index - 13
                end = cleavage_index + 2
                
                # Handle sequences that are shorter than the window or have cleavage sites near the ends
                if start < 0:
                    start = 0
                if end > len(sequence):
                    end = len(sequence)
                
                # Extract the window sequence
                window_seq = sequence[start:end]
                
                '''
                # If the extracted window is shorter than expected, pad it with gaps ('-')
                expected_length = 2 * window_size
                actual_length = len(window_seq)
                if actual_length < expected_length:
                    window_seq = window_seq.ljust(expected_length, '-')
                '''
                
                # Append the window sequence to the list
                self.aligned_sequences.append(window_seq)
                
                # Write the sequence to the FASTA file with a header
                fasta_file.write(f">{seq_id}\n{window_seq}\n")
        
        print(f"Aligned sequences extracted and written to {output_file}. Total aligned sequences: {len(self.aligned_sequences)}")
        
# ----------------------------
# Main Controller
# ----------------------------

class MainController:
    """
    Main controller class that manages user interactions and orchestrates the workflow.
    """

    def __init__(self, base_path_url: str):
        """
        Initializes the MainController.

        :param base_path_url: Base directory path where data files are located.
        """
        self.base_path_url = base_path_url

        self.show_menu = True

        self.positive_file_parser = RepresentativeFileParser(
            input_fasta=f"{base_path_url}/cluster-results_pos_rep_seq.fasta",
            output_tsv=f"{base_path_url}/positive_representative.tsv",
            original_tsv=f"{base_path_url}/positives.tsv",
            split_tsv=f"{base_path_url}/split_pos"
        )
        self.negative_file_parser = RepresentativeFileParser(
            input_fasta=f"{base_path_url}/cluster-results_neg_rep_seq.fasta",
            output_tsv=f"{base_path_url}/negative_representative.tsv",
            original_tsv=f"{base_path_url}/negatives.tsv",
            split_tsv=f"{base_path_url}/split_neg"
        )
        self.choices = ["A", "B", "C", "D", "E", "T", "F", "G", "H", "Q"]

    def display_menu(self):
        """
        Displays the menu options to the user, indicating the status of each step.
        """
        print("\n\nMENU\n")

        # Check status of each step by verifying the existence of necessary files
        if os.path.isfile(f"{self.base_path_url}/positives.fasta") and os.path.isfile(f"{self.base_path_url}/positives.tsv"):
            print("A) Fetch Data and Process - DONE")
        else:
            self.show_menu = False
            print("A) Fetch Data and Process - NOT DONE")

        if os.path.isfile(f"{self.base_path_url}/cluster-results_pos_rep_seq.fasta") and \
           os.path.isfile(f"{self.base_path_url}/cluster-results_neg_rep_seq.fasta"):
            print("B) Cluster with MMSeqs - DONE")
        else:
            self.show_menu = False
            print("B) Cluster with MMSeqs - NOT DONE")

        if os.path.isfile(f"{self.base_path_url}/positive_representative.tsv") and \
           os.path.isfile(f"{self.base_path_url}/negative_representative.tsv"):
            print("C) Create Files with Representative - DONE")
        else:
            self.show_menu = False
            print("C) Create Files with Representative - NOT DONE")

        if self.show_menu != False:
            self.choices = ["A", "B", "C", "D", "E", "T", "F", "G", "H", "Q"]
            print("D) Plot Data and Distributions of Protein Lengths")
            print("E) Plot Data and Distributions of Protein SP Lengths")
            print("F) Plot Amino Acid Frequencies against Uniprot Distribution")
            print("G) Plot Kingdom Pie Chart")
            print("H) Create Sequence Logo")
            print("Q) QUIT\n")
        else: 
            self.choices = ["A","B","C"]

    def handle_choice_a(self):
        """
        Handles Choice A: Fetch and process data.
        Fetches positive and negative datasets, processes them, and creates TSV and FASTA files.
        """
        # Define UniProt API URLs for positive and negative datasets
        base_url_positive = (
            "https://rest.uniprot.org/uniprotkb/search?format=json&query="
            "%28%28%28taxonomy_id%3A2759%29+NOT+%28fragment%3Atrue%29%29+AND+"
            "%28%28ft_signal_exp%3A*%29+AND+%28reviewed%3Atrue%29+AND+"
            "%28existence%3A1%29+AND+%28reviewed%3Atrue%29+AND+"
            "%28length%3A%5B40+TO+1000000000%5D%29%29%29"
        )
        base_url_negative = (
            "https://rest.uniprot.org/uniprotkb/search?format=json&query="
            "%28%28taxonomy_id%3A2759%29+AND+%28NOT+%28fragment%3Atrue%29+AND+"
            "NOT+%28ft_signal%3A*%29%29+AND+%28%28reviewed%3Atrue%29+AND+"
            "%28existence%3A1%29+AND+%28reviewed%3Atrue%29+AND+"
            "%28length%3A%5B40+TO+1000000000%5D%29+AND+"
            "%28%28cc_scl_term_exp%3ASL-0091%29+OR+%28cc_scl_term_exp%3ASL-0209%29+OR+"
            "%28cc_scl_term_exp%3ASL-0204%29+OR+%28cc_scl_term_exp%3ASL-0039%29+OR+"
            "%28cc_scl_term_exp%3ASL-0191%29+OR+%28cc_scl_term_exp%3ASL-0173%29%29%29%29"
        )

        # Define output file paths
        output_file_positive = f"{self.base_path_url}/positives"
        output_file_negative = f"{self.base_path_url}/negatives"

        # Process positive dataset
        print("Fetching and processing positive dataset...")
        positive_processor = PositiveUniProtEntryProcessor()
        positive_fetcher = UniProtDatasetFetcher(base_url_positive, positive_processor)
        positive_generator = positive_fetcher.fetch_dataset()

        positive_file_creator = UniProtFileCreator(output_file_positive)
        positive_file_creator.create_tsv_file(positive_generator, positive_processor)

        # Process negative dataset
        print("Fetching and processing negative dataset...")
        negative_processor = NegativeUniProtEntryProcessor()
        negative_fetcher = UniProtDatasetFetcher(base_url_negative, negative_processor)
        negative_generator = negative_fetcher.fetch_dataset()

        negative_file_creator = UniProtFileCreator(output_file_negative)
        negative_file_creator.create_tsv_file(negative_generator, negative_processor)

    def handle_choice_b(self):
        """
        Handles Choice B: Cluster sequences using MMSeqs.
        Performs clustering on positive and negative FASTA files.
        """
        cluster_manager = ClusterManager(base_path=self.base_path_url)

        # Cluster positive sequences
        print("Clustering positive sequences with MMSeqs...")
        cluster_manager.run_mmseqs_cluster(
            input_fasta=f"{self.base_path_url}/positives.fasta",
            cluster_output=f"{self.base_path_url}/cluster-results_pos",
            min_seq_id=0.3,
            coverage=0.4,
            cluster_mode=1
        )

        # Cluster negative sequences
        print("Clustering negative sequences with MMSeqs...")
        cluster_manager.run_mmseqs_cluster(
            input_fasta=f"{self.base_path_url}/negatives.fasta",
            cluster_output=f"{self.base_path_url}/cluster-results_neg",
            min_seq_id=0.3,
            coverage=0.4,
            cluster_mode=1
        )

    def handle_choice_c(self):
        """
        Handles Choice C: Create representative files.
        Extracts representative sequences and creates split datasets.
        """
        # Process positive representative sequences
        print("Processing positive representative sequences...")
        self.positive_file_parser.get_unique_element_information()
        self.positive_file_parser.print_representative()
        self.positive_file_parser.create_split_tsv()

        # Process negative representative sequences
        print("Processing negative representative sequences...")
        self.negative_file_parser.get_unique_element_information()
        self.negative_file_parser.print_representative()
        self.negative_file_parser.create_split_tsv()

    def handle_choice_d(self):
        """
        Handles Choice D: Plot data and distributions of protein lengths.
        Analyzes and plots sequence length distributions for training and benchmarking sets.
        """
        # Load training and benchmarking datasets
        positive_df = self.positive_file_parser.create_split_tsv()
        negative_df = self.negative_file_parser.create_split_tsv()

        # Analyze training set
        print("Analyzing training set sequence lengths...")
        training_analyzer = SequenceLengthAnalyzer(
            positive_df=positive_df,
            negative_df=negative_df,
            set_type="T",
            column_name="Seq_len"
        )
        training_analyzer.analyze()

        # Analyze benchmarking set
        print("Analyzing benchmarking set sequence lengths...")
        benchmarking_analyzer = SequenceLengthAnalyzer(
            positive_df=positive_df,
            negative_df=negative_df,
            set_type="B",
            column_name="Seq_len"
        )
        benchmarking_analyzer.analyze()

    def handle_choice_e(self):
        """
        Handles Choice E: Plot data and distributions of protein SP lengths.
        Analyzes and plots signal peptide length distributions for training and benchmarking sets.
        """
        # Load training and benchmarking datasets
        positive_df = self.positive_file_parser.create_split_tsv()
        #negative_df = self.negative_file_parser.create_split_tsv()

        # Analyze training set SP lengths
        print("Analyzing training set SP lengths...")
        training_sp_analyzer = SignalPeptideLengthAnalyzer(
            positive_df=positive_df,
            set_type="T",
            column_name="Cleavage_pos"
        )
        training_sp_analyzer.analyze()

        # Analyze benchmarking set SP lengths
        print("Analyzing benchmarking set SP lengths...")
        benchmarking_sp_analyzer = SignalPeptideLengthAnalyzer(
            positive_df=positive_df,
            set_type="B",
            column_name="Cleavage_pos"
        )
        benchmarking_sp_analyzer.analyze()

    def handle_choice_f(self):
        """
        Handles Choice F: Plot Amino Acid Frequencies.
        Analyzes and plots the average frequency of amino acids in the training set.
        """
        # Define the list of standard amino acids
        amino_acids = list(IUPACData.protein_letters)

        # Paths to split_pos.tsv and positives.fasta
        split_tsv = f"{self.base_path_url}/split_pos.tsv"
        fasta_file = f"{self.base_path_url}/positives.fasta"

        # Initialize the AminoAcidFrequencyAnalyzer
        aa_frequency_analyzer = AminoAcidFrequencyAnalyzer(
            split_tsv=split_tsv,
            fasta_file=fasta_file,
            amino_acids=amino_acids,
            set_type='T',  # Analyzing the Training set
            max_seq_length = 2000
        )

        # Perform the analysis and generate the plot
        aa_frequency_analyzer.analyze()

    def handle_choice_g(self):
        
        #ADD ALSO NEGATIVE -> make a new set to plot this

        positive_tsv_file_path = f"{self.base_path_url}/split_pos.tsv"
        negative_tsv_file_path = f"{self.base_path_url}/split_neg.tsv"

        analyzer = KingdomDistributionAnalyzer(positive_tsv_file_path, negative_tsv_file_path)
        analyzer.run_analysis()
    
    def handle_choice_h(self):
        # Initialize the class with file paths
        logo_maker = MakeSequenceLogo(
            base_path_url = self.base_path_url,
            positive_tsv_file_path="/split_pos.tsv",
            fasta_file_path="/positives.fasta"
        )
        
        # Load data
        logo_maker.load_data()

        logo_maker.load_fasta_sequences()

        # Create .fasta file
        logo_maker.extract_aligned_windows()        

    def handle_choice_q(self):
        """
        Handles Choice Q: Quit the program.
        """
        print("Exiting the program. Goodbye!")
        exit()

    def run(self):
        """
        Runs the main loop, displaying the menu and handling user choices.
        """
        while True:
            self.display_menu()
            choice = input("Enter your choice: ").strip().upper()
            print("\n")

            if choice not in self.choices:
                print("WRONG CHOICE, try again\n")
                continue

            if choice == "A":
                self.handle_choice_a()
            elif choice == "B":
                self.handle_choice_b()
            elif choice == "C":
                self.handle_choice_c()
            elif choice == "D":
                self.handle_choice_d()
            elif choice == "E":
                self.handle_choice_e()
            elif choice == "F":
                self.handle_choice_f()
            elif choice == "G":
                self.handle_choice_g()
            elif choice == "H":
                self.handle_choice_h()
            elif choice == "Q":
                self.handle_choice_q()

            else:
                print("Invalid choice, please try again.\n")

if __name__ == "__main__":

    #save images
    # plt.savefig('amino_acid_frequencies_comparison.png', dpi=300, bbox_inches='tight') before plt.show()

    # Define the base path where all data files are located
    base_path_url = f"../files"    #    base_path_url = "/Users/gianlucapiccolo/Desktop/lab2_2024/script/files/datasets"

    import os

    # This will give you the path to the directory just above the current directory
    parent_dir_path = os.path.join(os.path.dirname(__file__), os.pardir)

    # If you want to navigate into the "files" directory in the parent directory:
    base_path_url = os.path.join(parent_dir_path, "files")

    # You can also normalize the path using os.path.abspath to get the full path:
    parent_dir_path = os.path.abspath(parent_dir_path)
    base_path_url = os.path.abspath(base_path_url)

    # Initialize the main controller
    controller = MainController(base_path_url=base_path_url)

    # Run the main loop
    controller.run()

    #merge = MergeSplitDatasets(base_path_url,f"{base_path_url}/split_pos.tsv",f"{base_path_url}/split_neg.tsv",
                                #f"{base_path_url}/positives.fasta",f"{base_path_url}/negatives.fasta")
    
    #merge.addSequenceToTSV()