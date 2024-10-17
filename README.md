This script is designed to fetch, process, and analyze protein sequence data from the UniProt database. It includes functionalities for data fetching, processing, clustering, and various analyses such as sequence length distributions, signal peptide lengths, amino acid frequencies, and kingdom distribution. The script is modular, with classes handling specific tasks, and includes a main controller that orchestrates the workflow based on user input.

Prerequisites

    •	Python Version: 3.6 or higher
    •	External Tools:
    •	MMSeqs2: For sequence clustering (must be installed and accessible in the system path)
    •	Python Packages (specified in requirements.txt):
    •	numpy
    •	pandas
    •	requests
    •	seaborn
    •	biopython
    •	matplotlib

Overview of the Code

The script is organized into several sections and classes:

    1.	Data Fetching and Processing: Classes to fetch data from UniProt and process entries.
    2.	File Handling: Classes to create and manipulate TSV and FASTA files.
    3.	Clustering: Classes to perform sequence clustering using MMSeqs2.
    4.	Data Analysis: Classes to perform various analyses and generate plots.
    5.	Main Controller: A class that provides a menu-driven interface for the user to interact with the script.

How to Use the Code

    0. Create virtual environment if needed:
        python3 -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate

    1.	Install Required Packages:
        pip install -r requirements.txt

    2.	Install MMSeqs2:
        •	Download and install MMSeqs2 from MMSeqs2 GitHub.
        •	Ensure mmseqs is accessible from the command line.

    3.	Set the Base Path:
        Update the base_path_url variable in the script to point to the directory where data files will be stored.

            base_path_url = "/path/to/your/data/directory"

    4. Run the script:
        python data_fetch.py

    5.	Interact with the Menu:
    •	The script provides a menu with options labeled from A to H and Q to quit.
    •	Choose an option by entering the corresponding letter.
