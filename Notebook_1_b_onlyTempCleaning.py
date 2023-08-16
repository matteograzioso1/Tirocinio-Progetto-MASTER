# %% [markdown]
# # MASTER - Notebook 1 - Only temporal Cleaning
# ### Matteo Grazioso 884055

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

import myfunctions as mf # Custom functions

# %%
# Disply all columns and all rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%
# Find all txt files in the data folder
txt_files = mf.find_txt_files("data/raw")

# Choose a dataset from the list of txt files
selected_dataset = mf.choose_dataset(txt_files)

if selected_dataset:
    print(f"You selected the dataset {selected_dataset}")
else:
    print("No dataset selected.")

path  = selected_dataset

# %%
# The file contain the data of the validation of tickets in the city of public transport of Venice.

# Import the data into a dataframe of a txt file 
# path = 'data/raw/1-validazioni.txt'                     # Period: 2022-05-13 to 2022-07-15
# path = 'data/raw/2-esportazioneCompleta.txt'            # Period: 2023-01-23 to 2023-03-14
# path = 'data/raw/3-esportazionePasqua23.txt'            # Period: 2023-04-04 to 2023-06-03

df = pd.read_csv(path, header=0, sep='\t')
# Save the name of the file in a variable for future use extracting the name of the file from the path
file_name = path.split('/')[-1].split('.')[0]
# Remove the number and the - character from the head of the file name
file_name = file_name [file_name.find('-')+1:]


# %%
file_name

# %%
# Check the first 5 rows of the data
df.head()

# %%
# Check the last 5 rows of the data
df.tail()

# %%
# Create a subset of the data with the first 10% of the rows and the last 10% of the rows
# df = df.iloc[:int(len(df)*0.01),:]
# df = df.append(df.iloc[-int(len(df)*0.1):,:])


# %% [markdown]
# ## Explorative Data Analysis
# 

# %%
# Dates and hour of the validation of the ticket are in the same column 'DATA_VALIDAZIONE'
# Split the column 'DATA_VALIDAZIONE' into two columns 'DATA' and 'ORA' and convert them to datetime format
df.insert(0, 'DATA', pd.to_datetime(df['DATA_VALIDAZIONE'].str.split(' ').str[0], format='%d/%m/%Y'))
df.insert(1, 'ORA', pd.to_datetime(df['DATA_VALIDAZIONE'].str.split(' ').str[1], format='%H:%M').dt.time)

# Drop the column 'DATA_VALIDAZIONE'
# df.drop('DATA_VALIDAZIONE', axis=1, inplace=True)

# Display the first 5 rows of the dataframe
df.head()

# %%
# Set the format of the timestamp
df['DATA_VALIDAZIONE'] = pd.to_datetime(df['DATA_VALIDAZIONE'], format='%d/%m/%Y %H:%M')

# %%
# Print the date of the first and last validation using both data and hour
print('First validation: ', df['DATA'].min(), df['ORA'].min())
print('Last validation: ', df['DATA'].max(), df['ORA'].max())

# Print the number of Serial numbers
print('Number of Serial numbers: ', df['SERIALE'].nunique())

# Print the number of validation (rows)
print('Number of validation: ', df.shape[0])

# Print the number of tickets
print('Number of tickets: ', df['DESCRIZIONE_TITOLO'].nunique())
# Print the number of titolo
print('Number of titolo: ', df['TITOLO'].nunique())
# TODO: why the number of unique TITOLO is different from the number of DESCRIZIONE_TITOLO?

# Print the number of FERMATA
print('Number of FERMATA: ', df['FERMATA'].nunique())
# Print the number of DESCRIZIONE
print('Number of DESCRIZIONE: ', df['DESCRIZIONE'].nunique())
# TODO: why the number of unique DESCRIZIONE is different from the number of FERMATA?

# %%
# Which is the most used ticket?
df['DESCRIZIONE_TITOLO'].value_counts().head(10)

# %%
# Which is the most frequent validation in date and hour?
# Date and hour are in two different columns; DATA_VALIDAZIONE does not exist anymore
df.groupby(['DATA', 'ORA'])['SERIALE'].count().sort_values(ascending=False).head(10)
# TODO: #4 Re-aswer the question of the most frequent validation after cleaning operations

# %%
# Which is the most frequent FERMATA?
df['DESCRIZIONE'].value_counts().head(10)
# TODO: #4 Re-aswer the question of the most frequent FERMATA after cleaning operations

# %% [markdown]
# ## Categories

# %% [markdown]
# The column TICKET_CODE will be filled with the code of the ticket profile according to the ticket type and the ticket validity as follows:
# 
# **1.** One-day ticket
# 
# **2.** Two-day ticket
# 
# **3.** Three-day ticket
# 
# **4.** Weekly ticket (Seven-day ticket)
# 
# **5.** Monthly ticket
# 
# **5-STUD.** Monthly ticket for students
# 
# **5-RET.** Monthly ticket for retirees
# 
# **5-WKRS.** Monthly ticket for workers
# 
# **6.** Annual ticket
# 
# **6-STUD.** Annual ticket for students
# 
# **6-RET.** Annual ticket for retirees
# 
# **6-WKRS.** Annual ticket for workers
# 
# **7.** 75 minutes ticket
# 
# **8.** Other ticket (if it is necessary to add other types of tickets)

# %%
def assignTicketCode(df_a: pd.DataFrame):
    """
        This function assigns a ticket code to each row of the dataframe.
        :param df: the dataframe
        :return: the dataframe with the new column TICKET_CODE
    """
    # Add a new column with the code profile of the ticket
    df_a.insert(7, "TICKET_CODE", 'TBD')

    # Define the dictionary of ticket codes
    dict_tickets = {'1': 'One-day ticket', '2': 'Two-day ticket', '3': 'Three-day ticket', 
                '4': 'Seven-day ticket', 
                '5': 'Monthly ticket', '5-STUD': 'Monthly ticket for students',
                '5-RET': 'Monthly ticket for retired', '5-WKRS': 'Monthly ticket for workers',
                '6': 'Annual ticket', '6-STUD': 'Annual ticket for students', '6-RET': 'Annual ticket for retired',
                '6-WKRS': 'Annual ticket for workers',
                '7': '75 minutes ticket', '8': 'Other ticket'}
    
    # Convert the column 'DESCRIZIONE_TITOLO' into upper case 
    df_a['DESCRIZIONE_TITOLO'] = df_a['DESCRIZIONE_TITOLO'].str.upper()

    # One-day ticket
    df_a.loc[df_a['DESCRIZIONE_TITOLO'].str.contains('GIORNALIERO|24H|24ORE|24 ORE|DAILY'), 'TICKET_CODE'] = '1'
    # Two-day ticket
    df_a.loc[df_a['DESCRIZIONE_TITOLO'].str.contains('48H|48ORE|48 ORE'), 'TICKET_CODE'] = '2'
    # Three-day ticket
    df_a.loc[df_a['DESCRIZIONE_TITOLO'].str.contains('72H|72ORE|72 ORE'), 'TICKET_CODE'] = '3'
    # Seven-day ticket
    df_a[df_a['DESCRIZIONE_TITOLO'].str.contains('7GG|7DAYS|7 DAYS')]['DESCRIZIONE_TITOLO'].value_counts()
    # Monthly ticket
    df_a.loc[df_a['DESCRIZIONE_TITOLO'].str.contains('MENSILE|30GG|30 GG|MENS'), 'TICKET_CODE'] = '5'
    ## Monthly ticket for students
    df_a.loc[(df_a['TICKET_CODE'] == '5') & (df_a['DESCRIZIONE_TITOLO'].str.contains('STUDENTE|STUD')), 'TICKET_CODE'] = '5-STUD'
    ## Monthly ticket for retired
    df_a.loc[(df_a['TICKET_CODE'] == '5') & (df_a['DESCRIZIONE_TITOLO'].str.contains('OVER 65|65+|PENSIONATI')), 'TICKET_CODE'] = '5-RET'
    ## Monthly ticket for workers
    df_a.loc[(df_a['TICKET_CODE'] == '5') & (df_a['DESCRIZIONE_TITOLO'].str.contains('LAVORATORE|LAV')), 'TICKET_CODE'] = '5-WKRS'
    ## DDRG 1201-1297/2022
    df_a.loc[df_a['DESCRIZIONE_TITOLO'].str.contains('DDGR1201-1297/2022'), 'TICKET_CODE'] = '5'
    # Yearly ticket
    df_a.loc[df_a['DESCRIZIONE_TITOLO'].str.contains('ANNUALE|ANN|12MESI|12 MESI'), 'TICKET_CODE'] = '6'
    ## Yearly ticket for students
    df_a.loc[(df_a['TICKET_CODE'] == '6') & (df_a['DESCRIZIONE_TITOLO'].str.contains('STUDENTE|STUD|STUD')), 'TICKET_CODE'] = '6-STUD'
    ## Yearly ticket for retired
    df_a.loc[(df_a['TICKET_CODE'] == '6') & (df_a['DESCRIZIONE_TITOLO'].str.contains('OVER 65|65+|PENSIONATI')), 'TICKET_CODE'] = '6-RET'
    ## Yearly ticket for workers
    df_a.loc[(df_a['TICKET_CODE'] == '6') & (df_a['DESCRIZIONE_TITOLO'].str.contains('LAVORATORE|LAV|LAV')), 'TICKET_CODE'] = '6-WKRS'
    ## Student yearly ticket
    df_a.loc[(df_a['DESCRIZIONE_TITOLO'].str.contains('STUDENTE|STUD|STUD')) & ~ (df_a['TICKET_CODE'].isin(['5-STUD', '6-STUD'])), 'TICKET_CODE'] = '6-STUD'
    # 75 minutes ticket
    df_a.loc[df_a['DESCRIZIONE_TITOLO'].str.contains('75\'|75MIN|75 MIN'), 'TICKET_CODE'] = '7'
    # Other ticket
    df_a.loc[~df_a['TICKET_CODE'].isin(['1','2','3','4','5','5-STUD','5-WKRS','5-RET','6','6-STUD','6-WKRS','6-RET','7']), 'TICKET_CODE'] = '8'

    # Plot a pie chart of the column 'TICKET_CODE'
    fig, ax = plt.subplots(figsize=(20,10))
    df_a['TICKET_CODE'].value_counts().sort_index().plot.pie(startangle=90)

    # Add the name of the ticket profile on the pie chart
    plt.legend(labels=df_a['TICKET_CODE'].value_counts().sort_index().rename(dict_tickets).index, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=15)

    plt.title('Pie chart of the column TICKET_CODE', fontsize=20)
    plt.ylabel('')
    fig.patch.set_facecolor('white')
    plt.show()

    # Delete stamps with ticket code 8
    df_a = df_a[df_a['TICKET_CODE'] != '8']

    return df_a


# %%
df = assignTicketCode(df)
df.head()

# %% [markdown]
# ## Data Cleaning

# %% [markdown]
# ### Useless stamps

# %%
# Reset the index of the df and drop the old index in order to have a new index starting from 0 to the number of rows
# It is necessary to have a new index because the groupby function has created a multi-index
df.reset_index(drop=True, inplace=True)

# %%
# Create a new column 'MIN_TEMPORAL_GAP' that contains the minimum temporal gap between two validations for the same serial and fermata in minutes
df = df.groupby(['SERIALE','DATA', 'DESCRIZIONE']).apply(lambda x: x.assign(MIN_TEMPORAL_GAP = x['DATA_VALIDAZIONE'].diff().dt.total_seconds()/60))

# %%
df.head(20)

# %%
df.tail(20)

# %%
df['MIN_TEMPORAL_GAP'].value_counts()

# %%
# How many rows have a minimum temporal gap equal to NaN?
df[df['MIN_TEMPORAL_GAP'].isna()].shape[0]

# %%
# Prepare a file to save the information about the results of the cleaning process
# File txt with name: "cleaningResults + filename + date.txt"
import datetime
date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
with open('data/processed/cleaningResults_' + file_name + '_' + date + '.txt', 'w') as f:
    f.write('File name: ' + file_name + '\n')
    f.write('Operation starts at: ' + date + '\n')

# %%
# Cleaning operation: remove the rows using the minimum temporal gap

# Find a reasonable delta of MIN_TEMPORAL_GAP to remove the rows that have a minimum temporal gap for the same serial and fermata less than this delta
with open('data/processed/cleaningResults_' + file_name + '_' + date + '.txt', 'a') as f:
    # Print the minimum value of the column MIN_TEMPORAL_GAP
    print('The minimum value of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].min()))
    f.write('The minimum value of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].min()))

    # Print the maximum value of the column MIN_TEMPORAL_GAP
    print('The maximum value of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].max()))
    f.write('The maximum value of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].max()))

    # Print the mean value of the column MIN_TEMPORAL_GAP
    print('The mean value of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].mean()))
    f.write('The mean value of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].mean()))

    # Print the median value of the column MIN_TEMPORAL_GAP
    print('The median value of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].median()))
    f.write('The median value of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].median()))

    # Print the standard deviation of the column MIN_TEMPORAL_GAP
    print('The standard deviation of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].std()))
    f.write('The standard deviation of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].std()))

    # Print the 0.05th percentile of the column MIN_TEMPORAL_GAP
    print('The 0.05th percentile of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].quantile(0.05)))
    f.write('The 0.05th percentile of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].quantile(0.05)))

    # Print the 0.10th percentile of the column MIN_TEMPORAL_GAP
    print('The 0.10th percentile of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].quantile(0.10)))
    f.write('The 0.10th percentile of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].quantile(0.10)))

    # Print the 25th percentile of the column MIN_TEMPORAL_GAP
    print('The 25th percentile of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].quantile(0.25)))
    f.write('The 25th percentile of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].quantile(0.25)))

    # Print the 75th percentile of the column MIN_TEMPORAL_GAP
    print('The 75th percentile of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].quantile(0.75)))
    f.write('The 75th percentile of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].quantile(0.75)))

    # Print the 90th percentile of the column MIN_TEMPORAL_GAP
    print('The 90th percentile of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].quantile(0.90)))
    f.write('The 90th percentile of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].quantile(0.90)))

    # Print the 95th percentile of the column MIN_TEMPORAL_GAP
    print('The 95th percentile of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].quantile(0.95)))
    f.write('The 95th percentile of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].quantile(0.95)))

    # Print the 99th percentile of the column MIN_TEMPORAL_GAP
    print('The 99th percentile of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].quantile(0.99)))
    f.write('The 99th percentile of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].quantile(0.99)))

    # Print the 99.9th percentile of the column MIN_TEMPORAL_GAP
    print('The 99.9th percentile of the column MIN_TEMPORAL_GAP is: {}'.format(df['MIN_TEMPORAL_GAP'].quantile(0.999)))
    f.write('The 99.9th percentile of the column MIN_TEMPORAL_GAP is: {}\n'.format(df['MIN_TEMPORAL_GAP'].quantile(0.999)))

    # Decide the delta of MIN_TEMPORAL_GAP to remove the rows that have a minimum temporal gap for the same serial and fermata less than this delta
    delta = df['MIN_TEMPORAL_GAP'].quantile(0.1)
    if delta == 0:
        delta = df['MIN_TEMPORAL_GAP'].quantile(0.25)
    if delta == 0:
        delta = df['MIN_TEMPORAL_GAP'].median()
    print('The delta of MIN_TEMPORAL_GAP is: {}'.format(delta))
    f.write('The delta of MIN_TEMPORAL_GAP is: {}\n'.format(delta))

# %%
# Cleaning operation: remove the rows using the minimum temporal gap

# Save the number of rows before the cleaning operation
shape_before = df.shape[0]

# Delete the rows that have a minimum temporal gap for the same serial and fermata more than the delta calculated before.
# Do not remove the rows with NaN values because they are the first validations of the day of a specific serial and fermata usefull for the analysis
df = df[(df['MIN_TEMPORAL_GAP'] > delta) | (df['MIN_TEMPORAL_GAP'].isna())]

with open('data/processed/cleaningResults_' + file_name + '_' + date + '.txt', 'a') as f:
    # Print the number of rows before and after the cleaning operation and the difference
    print('The number of rows before the cleaning operation is: {}'.format(shape_before))
    f.write('The number of rows before the cleaning operation is: {}\n'.format(shape_before))
    print('The number of rows after the cleaning operation is: {}'.format(df.shape[0]))
    f.write('The number of rows after the cleaning operation is: {}\n'.format(df.shape[0]))
    print('The difference is: {}'.format(shape_before - df.shape[0]))
    f.write('The difference is: {}\n'.format(shape_before - df.shape[0]))

    # Calculate the percentage of rows that has just been deleted
    print('The percentage of rows that has just been deleted is: {}%'.format(round((shape_before - df.shape[0])/shape_before*100, 2)))
    f.write('The percentage of rows that has just been deleted is: {}%\n'.format(round((shape_before - df.shape[0])/shape_before*100, 2)))

# %%
# Delete the column MIN_TEMPORAL_GAP because it is not useful anymore
df.drop('MIN_TEMPORAL_GAP', axis=1, inplace=True)

# %%
# Create a new dataframe, copied from the original one
df_new = df.copy() 

# Print the head of the new dataframe
print(df_new.head())

# Export the new dataframe in a txt file
# The name of the file is dataset_cleaned followed by the name (file_name variable) of the file that has been cleaned with txt extension
name_file = 'dataset_cleaned_temp' + file_name.split('.')[0] + '.txt'
df_new.to_csv('data/processed/' + name_file, sep='\t', index=False)

print('The script has finished')
with open('data/processed/cleaningResults_' + file_name + '_' + date + '.txt', 'a') as f:
    f.write('The script has finished\n')
    f.write('The name of the file is: ' + name_file + '\n')
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write('Operation completed at: ' + date + '\n')


