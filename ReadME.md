# AI and Data Mining Techniques Applied to Public Transportation Analysis in Venice

This project is part of the MASTER initiative (Multiple ASpects TrajEctoRy management and analysis) and focuses on applying AI and data mining techniques to analyze public transportation in Venice, Italy. The project aims to understand user behavior and the flow of people throughout the city in collaboration with the primary public transport company, ACTV. This README provides an overview of the project and its objectives.

## Project Overview

The goal of this project is to study public transportation in Venice using AI and data mining techniques. By analyzing user validation stamps, which include date and time, user serial numbers (anonymized for privacy), code profiles, and stop codes, we create user trajectories that capture sequences of stops and corresponding times. We also categorize users based on their ticket profiles, such as one-day, two-day, three-day, seven-day, monthly, and yearly tickets.

## Objective

The primary objective of this project is to gain insights into user behavior and habits related to public transportation in Venice. By analyzing different datasets created for each ticket typology, we aim to understand tourist behavior as well as the routines of residents, workers, and students in Venice. The project aims to identify patterns, trends, and correlations in user behavior through data visualization techniques.

## Technologies Used

The project is developed using Python, leveraging various AI and data mining libraries and frameworks. The primary technologies and tools employed include:

- Python: The project is implemented using the Python programming language, providing flexibility, ease of use, and a rich ecosystem of libraries and tools.
- Data Mining Libraries: Popular Python libraries such as Pandas, NumPy, and Scikit-learn are used for data manipulation, analysis, and modeling.
- Visualization Libraries: Matplotlib and Seaborn are utilized to visualize the results and patterns extracted from the data.
- Machine Learning Algorithms: The project employs machine learning algorithms for data analysis, clustering, and classification tasks.

## Project Structure

The project is organized into several components:

1. Data Collection: User validation stamps are obtained from ACTV, the primary public transport company in Venice. This data includes timestamped user information and stop codes.

2. Data Preprocessing: The collected data is processed to create user trajectories by aggregating stop sequences and corresponding times. User profiles are also constructed based on ticket typologies.

3. Data Analysis: Various data mining techniques are applied to the user trajectories and profiles to identify patterns, trends, and correlations in user behavior. Machine learning algorithms may be employed for tasks such as clustering users or predicting user preferences.

4. Visualization: The results of the analysis are visualized using Matplotlib and Seaborn to extract meaningful insights and facilitate interpretation.


## Tree
- .gitattributes
- .gitignore
- .idea/
  - .gitignore
- Documenti/
  - Altro/
    - A BiLSTM-CNN model for predicting users’ next locations based on geotagged social media.pdf
    - ACTV_SAC2022-8.pdf
    - D5.2 - Approved From Portal.pdf
    - Tesi Filippo Zanatta.pdf
    - Tesi Francesco Andrea Antoniazzi.pdf
    - Transactions in GIS - 2018 - Chen - HiSpatialCluster  A novel high‐performance software tool for clustering massive spatial.pdf
    - legenda denominazioni titoli di viaggio.xlsx
  - Budspencer/
    - Notebook 1.b/
      - Notebook 1_b-Budspencer dataset 1.html
      - Notebook 1_b-Budspencer dataset 1.pdf
      - Notebook 1_b-Budspencer dataset 2.html
      - Notebook 1_b-Budspencer dataset 2.pdf
      - Notebook 1_b_dataset 3.html
      - Notebook 1_b_onlyTempCleaning_dataset 0.html
    - Notebook 2.i/
      - Notebook 2.i-Budspencer dataset 1.html
      - Notebook 2.i-Budspencer dataset 1.pdf
      - Notebook 2.i-Budspencer dataset 2.html
      - Notebook 2.i-Budspencer dataset 2.pdf
      - Notebook 2.i-Budspencer dataset 3.html
    - Notebook 2/
      - Notebook 2-Budspencer dataset 1.html
      - Notebook 2-Budspencer dataset 1.pdf
      - Notebook 2-Budspencer dataset 1.png
      - Notebook 2-Budspencer dataset 2.html
      - Notebook 2-Budspencer dataset 2.pdf
      - Notebook 2-Budspencer dataset 2.png
      - Notebook 2-Budspencer dataset 3.html
    - Report/
      - Report attività Master. rev. 1.0.pdf
      - Report attività Master. rev. 1.0.pptx
      - Report attività Master. rev. 1.1.pdf
      - Report attività Master. rev. 1.1.pptx
      - Report attività Master. rev. 1.2.pdf
      - Report attività Master. rev. 1.2.pptx
      - Report attività Master. rev. 1.3.pdf
      - Report attività Master. rev. 1.3.pptx
      - Report attività Master. rev. 1.4.pdf
      - Report attività Master. rev. 1.4.pptx
    - risultati clustering/
      - Nuovo Nuovo clustering/
        - Tabella delle differenze.pdf
        - Tabella delle differenze.xlsx
  - Report/
    - Report attività Master. rev. 1.0.pdf
    - Report attività Master. rev. 1.0.pptx
    - Report attività Master. rev. 1.1.pdf
    - Report attività Master. rev. 1.1.pptx
    - Report attività Master. rev. 1.2.pdf
    - Report attività Master. rev. 1.2.pptx
    - Report attività Master. rev. 1.3.pdf
    - Report attività Master. rev. 1.3.pptx
    - Report attività Master. rev. 1.4.pdf
    - Report attività Master. rev. 1.4.pptx
  - risultati clustering/
    - Nuovo Nuovo clustering/
      - Tabella delle differenze.pdf
      - Tabella delle differenze.xlsx
- MASTER/
  - README.md
  - requirements.txt
  - script/
    - clustering.ipynb
    - datasetInterfaces.ipynb
    - interfaces/
      - multipleDate/
        - code_mD.py
        - output.png
      - singleDate/
        - code_sd.py
        - output.png
      - trajectories/
        - code.py
        - code_t.py
        - outputMurano.png
        - outputSingleMurano.png
      - videoSingleDay/
        - code.py
        - code_v.py
        - output.png
        - outputVideo.mp4
- Notebook 1_a.ipynb
- Notebook 1_b.ipynb
- Notebook 1_b_onlyTempCleaning.ipynb
- Notebook 2.i .ipynb
- Notebook 2.ipynb
- Notebook 3 AUX GTFS.ipynb
- Notebook 3 AUX.ipynb
- Notebook 3 NEW.ipynb
- Notebook 3.ipynb
- Notebook 3AUX.ipynb
- Notebook 3B.ipynb
- Notebook 4 NUOVO.ipynb
- Notebook 4.ipynb
- Notebook3AUX.py
- Notebook4.py
- Notebook_1_b_onlyTempCleaning.py
- ReadME.md
- cc.py
- data/
  - dictionaries/
    - dict_prefix_Export.json
    - dict_prefix_esportazioneCompleta.json
    - dict_prefix_esportazionePasqua23.json
    - dict_prefix_validazioni.json
    - dict_ticket_codes.json
  - processed/
    - data-GTFS/
      - stop_aggr.json
      - stop_all.json
- file/
  - carnevale TURISTI UNITI.ipynb
  - carnevale.ipynb
  - clustering STUD-LOC-TUR.ipynb
  - clustering-LOC-TUR estate22.ipynb
  - estate22 TURISTI UNITI.ipynb
  - estate22.ipynb
  - functions.py
  - sistemazioneFileCsv.ipynb
- myfunctions.py
- requirements.txt
- stop_converted.json
