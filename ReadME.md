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
