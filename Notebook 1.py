#!/usr/bin/env python
# coding: utf-8

# # MASTER
# ### Matteo Grazioso 884055

# ## Categories

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Disply all columns and all rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


# The data is in the same directory as the notebook and is named 'validazioni.txt'. 
# The file contains the data of the validation of tickets in the city of public transport of Venice.

# Import the data into a dataframe of a txt file
df = pd.read_csv('validazioni.txt', header=0, sep='\t')


# In[ ]:


# Check the first 5 rows of the data
df.head()


# In[ ]:


# Add a new column with the code profile of the ticket
df.insert(6, "TICKET_CODE", 'TBD')

# This column will be filled with the code of the ticket profile according to the ticket type and the ticket validity as follows:
# 1. One-day ticket
# 2. Two-day ticket
# 3. Three-day ticket
# 4. Weekly ticket (Seven-day ticket)
# 5. Monthly ticket
# 6. Annual ticket
# 7. Other ticket (if it is necessary to add other types of tickets)


# In[ ]:


# Create a dictionary with the ticket code and the ticket profile
dict_tickets = {'1': 'One-day ticket', '2': 'Two-day ticket', '3': 'Three-day ticket', 
                '4': 'Seven-day ticket', '5': 'Monthly ticket', '6': 'Annual ticket',
                '7': 'Other ticket', '7a': 'A_Other ticket', '7b': 'B_Other ticket'}


# In[ ]:


# Which are the unique values of the column 'DESCRIZIONE_TITOLO'?
df['DESCRIZIONE_TITOLO'].unique()


# In[ ]:


# Get the number of unique values of the column 'DESCRIZIONE_TITOLO'
num_unique_DESCRIZIONE_TITOLO = len(df['DESCRIZIONE_TITOLO'].unique())
print('The number of unique values of the column DESCRIZIONE_TITOLO is: ', num_unique_DESCRIZIONE_TITOLO)


# In[ ]:


# How many tickets are there for each type?
# Print the count and the average of validation for each ticket type
df['DESCRIZIONE_TITOLO'].value_counts()


# ### One-day tickets

# In[ ]:


# Which type of ticket are one-day tickets and how many are there?
# Exclude the tickets that contains also 48h or 72h
df[df['DESCRIZIONE_TITOLO'].str.contains('giornaliero|24h|24 ore|24|Daily|daily|DAILY') & ~df['DESCRIZIONE_TITOLO'].str.contains('48|72')]['DESCRIZIONE_TITOLO'].value_counts()


# In[ ]:


# Populate the column 'TICKET_CODE' with the code of the ticket profile according to the ticket type and the ticket validity 
df.loc[df['DESCRIZIONE_TITOLO'].str.contains('giornaliero|24h|24 ore|24|Daily|daily|DAILY') & ~df['DESCRIZIONE_TITOLO'].str.contains('48|72'), 'TICKET_CODE'] = '1'


# ### Two days tickets

# In[ ]:


# Which type of ticket are two-day tickets and how many are there?
df[df['DESCRIZIONE_TITOLO'].str.contains('48')]['DESCRIZIONE_TITOLO'].value_counts()


# In[ ]:


# Populate the column 'TICKET_CODE' with the code of the ticket profile according to the ticket type and the ticket validity
df.loc[df['DESCRIZIONE_TITOLO'].str.contains('48'), 'TICKET_CODE'] = '2'


# ### Three days tickets

# In[ ]:


# Which type of ticket are three-day tickets and how many are there?
df[df['DESCRIZIONE_TITOLO'].str.contains('72')]['DESCRIZIONE_TITOLO'].value_counts()


# In[ ]:


# Populate the column 'TICKET_CODE' with the code of the ticket profile according to the ticket type and the ticket validity
df.loc[df['DESCRIZIONE_TITOLO'].str.contains('72'), 'TICKET_CODE'] = '3'


# ### Seven days tickets

# In[ ]:


# Which type of ticket are weekly tickets and how many are there?
# Exclude the tickets that contains also 72, 75 that are three-day tickets, 17, 48h, 57 that are other types of tickets and
# 'tratt*' and 'tr' that are reserved to specific routes
df[df['DESCRIZIONE_TITOLO'].str.contains('settimanale|7|Weekly') & ~df['DESCRIZIONE_TITOLO'].str.contains('72|75|17|48h|57|tratt*|tr')]['DESCRIZIONE_TITOLO'].value_counts()


# In[ ]:


# Populate the column 'TICKET_CODE' with the code of the ticket profile according to the ticket type and the ticket validity
df.loc[df['DESCRIZIONE_TITOLO'].str.contains('settimanale|7|Weekly') & ~df['DESCRIZIONE_TITOLO'].str.contains('72|75|17|48h|57|tratt*|tr'), 'TICKET_CODE'] = '4'


# ### Monthly tickets

# In[ ]:


# Whick type of ticket are monthly tickets and how many are there?
df[df['DESCRIZIONE_TITOLO'].str.contains('abbonamento|mensile|30')]['DESCRIZIONE_TITOLO'].value_counts()


# In[ ]:


# Populate the column 'TICKET_CODE' with the code of the ticket profile according to the ticket type and the ticket validity
df.loc[df['DESCRIZIONE_TITOLO'].str.contains('abbonamento|mensile|30'), 'TICKET_CODE'] = '5'


# ### Yearly tickets

# In[ ]:


# Which type of ticket are annual tickets and how many are there?
df[df['DESCRIZIONE_TITOLO'].str.contains('annuale|365|year|yearly')]['DESCRIZIONE_TITOLO'].value_counts()


# In[ ]:


# Populate the column 'TICKET_CODE' with the code of the ticket profile according to the ticket type and the ticket validity
df.loc[df['DESCRIZIONE_TITOLO'].str.contains('annuale|365|year|yearly'), 'TICKET_CODE'] = '6'


# ### 75 minutes tickets

# In[ ]:


# Which type of ticket are 75' (75 minutes) tickets and how many are there?
df[df['DESCRIZIONE_TITOLO'].str.contains('75')]['DESCRIZIONE_TITOLO'].value_counts()


# In[ ]:


# Populate the column 'TICKET_CODE' with the code of the ticket profile according to the ticket type and the ticket validity
df.loc[df['DESCRIZIONE_TITOLO'].str.contains('75'), 'TICKET_CODE'] = '7a'


# ### Other types of tickets

# In[ ]:


# Which type of ticket are other tickets and how many are there?
# The other tickets are the tickets that are not already classified in the previous categories
df[~df['TICKET_CODE'].isin(['1','2','3','4','5','6', '7a'])]['DESCRIZIONE_TITOLO'].value_counts()


# In[ ]:


# Populate the column 'TICKET_CODE' with the code of the ticket profile according to the ticket type and the ticket validity
df.loc[~df['TICKET_CODE'].isin(['1','2','3','4','5','6', '7a']), 'TICKET_CODE'] = '7b'


# ### Summary of the ticket profiles

# In[ ]:


# Print the number of tickets for each ticket profile code ordered by the code of the ticket profile; print the name of the ticket profile using the dictionary 'dict_tickets'
df['TICKET_CODE'].value_counts().sort_index().rename(dict_tickets).reindex(dict_tickets.values(), fill_value=0)


# In[ ]:


# Countplot of the column 'TICKET_CODE'
fig, ax = plt.subplots(figsize=(15,8))
# Countplot of the column 'TICKET_CODE'
sns.countplot(x='TICKET_CODE', data=df, order=df['TICKET_CODE'].value_counts().sort_index().index)
plt.title('Countplot of the column TICKET_CODE', fontsize=20)
plt.xlabel('Ticket code', fontsize=15)
plt.ylabel('Count (in millions)', fontsize=15)

# Change yticks to have a better visualization
scale = np.arange(0, max(df['TICKET_CODE'].value_counts())+100000, 100000)
plt.yticks(scale)

# Add the percentage of each category on top of the bars
for p in ax.patches:
    ax.annotate('{:.3f}%'.format(100*p.get_height()/len(df)), (p.get_x()+0.3, p.get_height()+10000))

# Add the count of each category on top of the bars
for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+30000))

# Add a padding on the top of the plot
plt.subplots_adjust(top=3)

plt.show()


# In[ ]:


# Plot a pie chart of the column 'TICKET_CODE'
fig, ax = plt.subplots(figsize=(20,10))
df['TICKET_CODE'].value_counts().sort_index().plot.pie(autopct='%1.3f%%', startangle=90, pctdistance=0.85, labeldistance=1.1)

# Add the name of the ticket profile on the pie chart
plt.legend(labels=df['TICKET_CODE'].value_counts().sort_index().rename(dict_tickets).index, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=15)

plt.title('Pie chart of the column TICKET_CODE', fontsize=20)
plt.ylabel('')
fig.patch.set_facecolor('white')
plt.show()


# #### Focus on Montly tickets

# In[ ]:


# Find the 'FERMATA' and 'DESCRIZIONE' distributions of the monthly tickets
df[df['TICKET_CODE'] == '5']['FERMATA'].value_counts().sort_index()

print('The number of different stops (FERMATA) where the monthly tickets are used is: {}'.format(len(df[df['TICKET_CODE'] == '5']['FERMATA'].value_counts().sort_index())))
print('The stops (FERMATA) where the monthly tickets are used are: {}'.format(df[df['TICKET_CODE'] == '5']['FERMATA'].value_counts().sort_index().index))
print('The descriptions of the stops (DESCRIZIONE) of the monthly tickets are: {}'.format(df[df['TICKET_CODE'] == '5']['DESCRIZIONE'].value_counts().sort_index().index))


# In[ ]:


# Plot a pie chart of the column 'DESCRIZIONE' of the monthly tickets
fig, ax = plt.subplots(figsize=(20,10))
df[df['TICKET_CODE'] == '5']['DESCRIZIONE'].value_counts().sort_index().plot.pie(autopct='%1.3f%%', startangle=90, pctdistance=0.85, labeldistance=1.1)

# Add the name of the ticket profile on the pie chart
plt.legend(labels=df[df['TICKET_CODE'] == '5']['DESCRIZIONE'].value_counts().sort_index().index, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=15)

plt.title('Pie chart of the column DESCRIZIONE of the monthly tickets', fontsize=20)
plt.ylabel('')

fig.patch.set_facecolor('white')
plt.show()


# In[ ]:


df.shape


# #### Delete the validation that are with TICKET_CODE = 7b (other tickets) 

# In[ ]:


shape_before = df.shape[0]

# Delete 7b tickets because they are not useful for the analysis 
df = df[df['TICKET_CODE'] != '7b']

# Print the number of rows before and after the deletion of the 7b tickets and the difference
print('The number of rows before the deletion of the 7b tickets is: {}'.format(shape_before))
print('The number of rows after the deletion of the 7b tickets is: {}'.format(df.shape[0]))
print('The difference is: {}'.format(shape_before - df.shape[0]))
print('b')


# ## Data Cleaning

# ### Useless stamps

# In[ ]:


# The timestamp is in the column 'DATA_VALIDAZIONE' in the format '%d/%m-%Y %H:%M'

# Convert the column 'DATA_VALIDAZIONE' to datetime
#df['DATA_VALIDAZIONE'] = pd.to_datetime(df['DATA_VALIDAZIONE'])

# Set the format of the column 'DATA_VALIDAZIONE' to '%d/%m/%Y %H:%M'
#df['DATA_VALIDAZIONE'] = df['DATA_VALIDAZIONE'].dt.strftime('%d/%m/%Y %H:%M')

# Print the head of the dataframe to see the new format of the column 'DATA_VALIDAZIONE'
#df.head()


# In[ ]:


# Print the shape of the dataframe
print('The shape of the dataframe is: {}'.format(df.shape))


# In[ ]:


# The timestamp is in the column 'DATA_VALIDAZIONE' in the format '%d/%m/%Y %H:%M'
# If there are more than one validation for the same user (SERIALE) in 2 minutes, then the validation is considered as a mistake so keep only the last validation for each user in 2 minutes

# Sort the dataframe by the column 'SERIALE' and the column 'DATA_VALIDAZIONE'
#df_sort = df.sort_values(by=['SERIALE', 'DATA_VALIDAZIONE'])

# Print the head of the dataframe to see the new order of the rows
#df_sort.head(20)


# In[ ]:


# Create a new column 'DATA_VALIDAZIONE_2' that is the column 'DATA_VALIDAZIONE' shifted by 1 row
#df_sort['DATA_VALIDAZIONE_2'] = df_sort['DATA_VALIDAZIONE'].shift(1)

# Print the head of the dataframe to see the new column 'DATA_VALIDAZIONE_2'
#df_sort.head(20)


# In[ ]:


# Create a new column 'SERIALE_2' that is the column 'SERIALE' shifted by 1 row
#df_sort['SERIALE_2'] = df_sort['SERIALE'].shift(1)

# Print the head of the dataframe to see the new column 'SERIALE_2'
#df_sort.head(20)


# In[ ]:


# fill the NaN values with 0
#df_sort['SERIALE_2'] = df_sort['SERIALE_2'].fillna(0)
#df_sort['DATA_VALIDAZIONE_2'] = df_sort['DATA_VALIDAZIONE_2'].fillna(0)


# In[ ]:


# Create a new column 'DIFF' that is the difference between the column 'DATA_VALIDAZIONE' and the column 'DATA_VALIDAZIONE_2'
#df_sort['DIFF'] = pd.to_datetime(df_sort['DATA_VALIDAZIONE']) - pd.to_datetime(df_sort['DATA_VALIDAZIONE_2'])

# Print the head of the dataframe to see the new column 'DIFF'
#df_sort.head(20)


# In[ ]:


# Create a new column 'DIFF_MINUTES' that is the difference between the column 'DATA_VALIDAZIONE' and the column 'DATA_VALIDAZIONE_2' in minutes
#df['DIFF_MINUTES'] = df['DIFF'].dt.total_seconds()/60

# Print the head of the dataframe to see the new column 'DIFF_MINUTES'
#df.head()


# In[ ]:


# If the difference between the column 'DATA_VALIDAZIONE' and the column 'DATA_VALIDAZIONE_2' is less than 2 minutes and the column 'SERIALE' is equal to the column 'SERIALE_2', then the validation is considered as a mistake
# So keep only the last validation for each user in 2 minutes
#df_to_drop = df[(df['DIFF_MINUTES'] < 2) & (df['SERIALE'] == df['SERIALE_2'])]
#df = df.drop(df[(df['DIFF_MINUTES'] < 2) & (df['SERIALE'] == df['SERIALE_2'])].index)


# ### Stops similar

# In[ ]:


# Print the number of unique values in the column 'DESCRIZIONE' that are the names of the stops
print('The number of unique values in the column DESCRIZIONE is: {}'.format(df['DESCRIZIONE'].nunique()))


# In[ ]:


# Define a function that returns the common prefix of a list of strings
def get_common_prefix(string_list):
    # input di tipo  string_list = ["Mestre Centro", "Mirano Centro"]
    first_prefix = string_list[0].split(" ")[0]
    # create and empty dictionary
    prefix_dict = {}

    for string in string_list[1:]:
        if not string.startswith(first_prefix):
            first_prefix = string.split(" ")[0]
            if string.startswith(first_prefix):
                # In the dictionary add the new prefix as key and the list of strings that have this prefix as value
                prefix_dict[first_prefix] = [string for string in string_list if string.startswith(first_prefix)]
        else:
            # In the dictionary add the new prefix as key and the list of strings that have this prefix as value
            prefix_dict[first_prefix] = [string for string in string_list if string.startswith(first_prefix)]
    return prefix_dict
    


# In[ ]:


# Use the function get_common_prefix to find the common prefix of the strings in the column 'DESCRIZIONE' and print the result

# Crete a string list with the unique values of the column 'DESCRIZIONE'
string_list = df['DESCRIZIONE'].unique().tolist()

dict_prefix = get_common_prefix(string_list)
for key, value in dict_prefix.items():
    print('{}: {}'.format(key, value))

# Print the number of keys in the dictionary
print('The number of keys in the dictionary is: {}'.format(len(dict_prefix.keys())))


# In[ ]:





# #### Update some keys in the dictionary

# In[ ]:


# Rename the key 'P.le' with 'P.le Roma'
dict_prefix['P.le Roma'] = dict_prefix.pop('P.le')
# Rename the key 'F.TE' with 'F.TE NOVE'
dict_prefix['F.TE NOVE'] = dict_prefix.pop('F.TE')


# In[ ]:


# Print the values of the dictionary with the keys 'S.' and 'San'
print('The values of the dictionary with the key S. are: {}'.format(dict_prefix['S.']))
print('The values of the dictionary with the key San are: {}'.format(dict_prefix['San']))


# ##### S.Erasmo

# In[ ]:


# Create a new key in the dictionary with the key S.ERASMO; insert as value the list of strings that have the prefix 'S.ERASMO'
dict_prefix['S.ERASMO'] = [string for string in dict_prefix['S.'] if string.startswith('S.ERASMO')]

# Add the value 'S. Erasmo Pu' originally in the key 'San' to the key 'S.ERASMO'
dict_prefix['S.ERASMO'].append('S. Erasmo Pu')

# Remove the strings that have the prefix 'S.ERASMO' from the keys 'S.' and 'San'
dict_prefix['S.'] = [string for string in dict_prefix['S.'] if not string.startswith('S.ERASMO')]
dict_prefix['S.'] = [string for string in dict_prefix['S.'] if not string.startswith('S. Erasmo Pu')]

# Print the values of the dictionary with the key 'S.ERASMO'
print('The values of the dictionary with the key S.ERASMO are: {}'.format(dict_prefix['S.ERASMO']))


# ##### San Marco

# In[ ]:


# Create a new key in the dictionary with the key 'San Marco'; insert as value the list of strings that have the prefix 'San Marco'
dict_prefix['San Marco'] = [string for string in dict_prefix['San'] if string.startswith('San Marco')]


# Add the value S. MARCO (Gi', 'S. Pietro in Gu') originally in the key 'S.' to the key 'San Marco'
dict_prefix['San Marco'].append('S. MARCO (Gi')

# Remove the strings that have the prefix 'San Marco' from the keys 'S.' and 'San'
dict_prefix['San'] = [string for string in dict_prefix['San'] if not string.startswith('San Marco')]
dict_prefix['S.'] = [string for string in dict_prefix['S.'] if not string.startswith('S. MARCO (Gi')]

# Print the values of the dictionary with the key 'San Marco'
print('The values of the dictionary with the key San Marco are: {}'.format(dict_prefix['San Marco']))


# ##### San Dona'

# In[ ]:


# Create a new key in the dictionary with the key 'San Dona'; insert as value the list of strings that have the prefix 'San Dona'
dict_prefix['San Dona'] = [string for string in dict_prefix['San'] if string.startswith('San Dona')]

# Remove the strings that have the prefix 'San Dona' from the keys 'S.' and 'San'
dict_prefix['San'] = [string for string in dict_prefix['San'] if not string.startswith('San Dona')]

# Print the values of the dictionary with the key 'San Dona'
print('The values of the dictionary with the key San Dona are: {}'.format(dict_prefix['San Dona']))


# #### San Pietro

# In[ ]:


# Create a new key in the dictionary with the key 'San Pietro'; insert as value the list of strings that have the word 'Pietro' in the string
dict_prefix['San Pietro'] = [string for string in dict_prefix['San'] if 'Pietro' in string] + [string for string in dict_prefix['S.'] if 'Pietro' in string]

# Remove the strings that have the word 'Pietro' from the keys 'S.' and 'San'
dict_prefix['San'] = [string for string in dict_prefix['San'] if 'Pietro' not in string]
dict_prefix['S.'] = [string for string in dict_prefix['S.'] if 'Pietro' not in string]

# Print the values of the dictionary with the key 'San Pietro'
print('The values of the dictionary with the key San Pietro are: {}'.format(dict_prefix['San Pietro']))


# #### Ca' Rossa

# In[ ]:


# Create a new key in the dictionary with the key 'Ca' Rossa'; insert as value the list of strings that have the word 'Ca' Rossa' in the string
dict_prefix['Ca\' Rossa'] = [string for string in dict_prefix['Ca\''] if 'Ca' in string and 'Rossa' in string]

# Remove the strings that have the word 'Ca' Rossa' from the keys 'Ca''
dict_prefix['Ca\''] = [string for string in dict_prefix['Ca\''] if 'Ca' not in string or 'Rossa' not in string]

# Print the values of the dictionary with the key 'Ca Rossa'
print('The values of the dictionary with the key Ca\' Rossa are: {}'.format(dict_prefix['Ca\' Rossa']))



# ##### Manage the remaining values in the keys 'S.' and 'San' and others

# In[ ]:


# Manage the remaining values in the keys 'S.', 'San', 'Santa', 'Sant'', 'Ca'', 'Piazza', 'Piazzale', 'Stazione', 'Treviso, 'Trento', 'Incr.'
# Create a new key for each value in the keys as above and assign the value as value of the new key
for value in dict_prefix['S.']:
    dict_prefix[value] = [value]

for value in dict_prefix['San']:
    dict_prefix[value] = [value]

for value in dict_prefix['Santa']:
    dict_prefix[value] = [value]

for value in dict_prefix['Sant\'']:
    dict_prefix[value] = [value]

for value in dict_prefix['Ca\'']:
    dict_prefix[value] = [value]

for value in dict_prefix['Piazza']:
    dict_prefix[value] = [value]

for value in dict_prefix['Piazzale']:
    dict_prefix[value] = [value]

for value in dict_prefix['Stazione']:
    dict_prefix[value] = [value]

for value in dict_prefix['Treviso']:
    dict_prefix[value] = [value]

for value in dict_prefix['Trento']:
    dict_prefix[value] = [value]

for value in dict_prefix['Incr.']:
    dict_prefix[value] = [value]



# Remove the keys 'S.' and 'San' witout printing the values
dict_prefix.pop('S.')
dict_prefix.pop('San')
dict_prefix.pop('Santa')
dict_prefix.pop('Sant\'')
dict_prefix.pop('Ca\'')
dict_prefix.pop('Piazza')
dict_prefix.pop('Piazzale')
dict_prefix.pop('Stazione')
dict_prefix.pop('Treviso')
dict_prefix.pop('Trento')
dict_prefix.pop('Incr.')


# #### Treviso and Trento

# In[ ]:


# Create a new key in the dictionary with the keys 'Treviso' and 'Trento'; 
# insert as value the list of strings that have the word 'Treviso' or 'Trento' in the string
dict_prefix['Treviso'] = [string for string in dict_prefix['Tre'] if 'Treviso' in string]
dict_prefix['Trento'] = [string for string in dict_prefix['Tre'] if 'Trento' in string]

# Remove the strings that have the word 'Treviso' or 'Trento' from the dictionary
dict_prefix['Treviso'] = [string for string in dict_prefix['Tre'] if 'Treviso' not in string]
dict_prefix['Trento'] = [string for string in dict_prefix['Tre'] if 'Trento' not in string]

# Print the values of the dictionary with the key 'Treviso'
print('The values of the dictionary with the key Treviso are: {}'.format(dict_prefix['Treviso']))


# TODO: Correct the values of the keys 'Treviso' and 'Trento' with the correct values: PORCO D


# #### Keys with only an item

# In[ ]:


# If a key as only one value, then rename the key with the value
# Use copy() to avoid RuntimeError: dictionary changed size during iteration
for key, value in dict_prefix.copy().items():
    if len(value) == 1:
        dict_prefix[value[0]] = dict_prefix.pop(key)


# ##### Finally, the update dictionary is

# In[ ]:


# Print the dictionary in the new format
for key, value in dict_prefix.items():
    print('{}: {}'.format(key, value))


# In[ ]:


# Export the dictionary in a json file
import json
with open('dict_prefix.json', 'w') as fp:
    json.dump(dict_prefix, fp)


# In[ ]:


# Create a new dataframe, copied from the original one
df_new = df.copy() 

# Update the column 'DESCRIZIONE' of the new df with the new values of the dictionary: 
# the value that are present in the dataframe are the values of the dictionary; you have to sobstitute with the key of the dictionary
for key, value in dict_prefix.items():
    df_new['DESCRIZIONE'] = df_new['DESCRIZIONE'].replace(value, key)


# Print the head of the new dataframe
print(df_new.head())

# Export the new dataframe in a txt file
df_new.to_csv('df_new.txt', sep='\t', index=False)

print('The script has finished')

