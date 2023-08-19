# %% [markdown]
# # MASTER - Notebook 3 AUX
# ## Gets geocoordinates for each stop and create a new dataset.
# ### Matteo Grazioso 884055

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime
import re

from pandas import Timestamp
import json
import warnings
warnings.filterwarnings('ignore')

import datetime

import myfunctions as mf # Custom functions

# %%
# Disply all columns and all rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%
# Find all txt files in the data folder
txt_files = mf.find_txt_files("data/processed/")

# Choose a dataset from the list of txt files
selected_dataset = mf.choose_dataset(txt_files)

if selected_dataset:
    print(f"You selected the dataset {selected_dataset}")
else:
    print("No dataset selected.")

path  = selected_dataset

# %%
# The file contains the data of the validation of tickets in the city of public transport of Venice.
# The file has been created by the Notebook 1.ipynb

df = pd.read_csv(path, header=0, sep='\t')

# Save the name of the file in a variable for future use extracting the name of the file from the path
file_name = path.split('_')[-1].split('.')[0]
# if path contains the word 'temp' then append the word 'temp' to the name of the file
if 'temp' in path:
    file_name = 'temp_' + file_name
subfolder = file_name

# Display the first 5 rows of the dataframe
df.head()

# Convert the column 'DATA' to datetime format
df['DATA'] = pd.to_datetime(df['DATA'], format='%Y-%m-%d')

# Take only the first 1%
# df = df.iloc[:int(len(df) * 0.01)]

# %%
# Print teh range of dates
print(f"Date range: {df['DATA'].min()} - {df['DATA'].max()}")

# %%
note = ""
# If data range is 04-02-2023 to 21-02-2023, then note = carnival_period
# If data range is 04-03-2023 to 21-03-2023, then note = after_carnival_period
# If dara range is 17-01-2023 to 03-02-2023, then note = before_carnival_period

# Get data range
start_date = df['DATA'].min()
end_date = df['DATA'].max()
# Check if data range is carnival_period or after_carnival_period
if start_date == datetime.date(2023, 2, 4) and end_date == datetime.date(2023, 2, 21):
    note = "carnival_period"
elif start_date == datetime.date(2023, 3, 4) and end_date == datetime.date(2023, 3, 21):
    note = "after_carnival_period"
elif start_date == datetime.date(2023, 1, 17) and end_date == datetime.date(2023, 2, 3):
    note = "before_carnival_period"

print("Note: ", note)

# %%
if path == 'data/processed/dataset_cleaned_esportazionePasqua23.txt':
    # Divide the dataset in two parts but be sure to not divide the same day in two parts; the size of the two parts is the same
    df1 = df[df['DATA'] < '2023-05-04']
    df2 = df[df['DATA'] >= '2023-05-04']

    # Check the size of the two parts also in percentage
    print(f"Size of the first part: {df1.shape}")
    print(f"Size of the second part: {df2.shape}")
    print(f"Size of the first part in percentage: {df1.shape[0]/df.shape[0]*100:.2f}%")
    print(f"Size of the second part in percentage: {df2.shape[0]/df.shape[0]*100:.2f}%")

    # Export the two parts in two different files
    import os
    if not os.path.exists(f"data/processed/{subfolder}"):
        os.makedirs(f"data/processed/{subfolder}")
    df1.to_csv(f"data/processed/{subfolder}/df1_{file_name}_divided.csv", sep='\t', index=False)
    df2.to_csv(f"data/processed/{subfolder}/df2_{file_name}_divided.csv", sep='\t', index=False)

# %%
def open_dict_trajectories(is_focus_on_ticket_code: bool = False, ticket_code: str = "", is_focus_on_tourists: bool = False, is_focus_on_workers: bool = False) -> dict:
    """
        This function opens the dictionary of trajectories eventually filtered by ticket code, tourists or workers.
        :param is_focus_on_ticket_code: if True, the dictionary will be filtered by ticket code
        :param ticket_code: the ticket code to filter the dictionary
        :param is_focus_on_tourists: if True, the dictionary will be filtered by tourists
        :param is_focus_on_workers: if True, the dictionary will be filtered by workers
        :return: the dictionary of trajectories eventually filtered
    """
    if is_focus_on_ticket_code:
        # Open the dictionary of trajectories filtered by ticket code
        with open('data/dictionaries/trajectories/' + subfolder + '/dict_trajectories_' + file_name + '_tc:' + ticket_code + '.json') as f:
            data = json.load(f)
        return data
    elif is_focus_on_tourists:
        # Open the dictionary of trajectories filtered by tourists
        with open('data/dictionaries/trajectories/' + subfolder + '/dict_trajectories_' + file_name + '_tourists.json') as f:
            data = json.load(f)
        return data
    elif is_focus_on_workers:
        # Open the dictionary of trajectories filtered by workers
        with open('data/dictionaries/trajectories/' + subfolder + '/dict_trajectories_' + file_name + '_workers.json') as f:
            data = json.load(f)
        return data
    else:
        # Open the dictionary of trajectories without filters
        with open('data/dictionaries/trajectories/' + subfolder + '/dict_trajectories_' + file_name + '.json') as f:
            data = json.load(f)
        return data

# %%
def extract_values_key(key: str) -> tuple:
    """
        This function extracts the values of the key.
        :param key: the key in the format (serial, day) where day can be None
        :return: the key in a tuple format (serial, day)
    """
    pattern = r"\((-?\d+),\s?(None|Timestamp\('.*'\))\)"
    match = re.match(pattern, key)
    if match:
        serial = int(match.group(1))
        # Timestamp is None if the second group is 'None', otherwise it is a Timestamp object not datetime
        timestamp = None if match.group(2) == 'None' else pd.Timestamp(match.group(2)[11:-2])
        return serial, timestamp
    else:
        # Print a messagge in red in a pretty format
        print('\033[91m' + 'The key {} is not in the correct format.'.format(key) + '\033[0m')
        return None, None


# %%
def get_rows_from_key(df_k: pd.DataFrame, key: tuple) -> pd.DataFrame:
    """
        This function returns the rows of the dataframe for the specified key.
        Note that the key is in the format (serial, day), where day can be None that means that the seriale doesn't change over the days.
        :param df_k: the dataframe
        :param key: the key
        :return: the row of the dataframe
    """
    # If the day is None, return the dataframe with the trajectories of the user
    if key[1] == None:
        print('The key is: {}'.format(key))
        return df_k[df_k['SERIALE'] == key[0]]
    # Otherwise, return the dataframe with the trajectories of the user in the specified day
    else:
        print('The key is: {}'.format(key))
        # Notice that the data in the dataframe is a string while the data in the key is a Timestamp
        # print('The key is: {}'.format(key))
        # Convert the data in the dataframe to a Timestamp
        df_k['DATA'] = pd.to_datetime(df_k['DATA'], format='%Y-%m-%d %H:%M:%S')
        return df_k[(df_k['SERIALE'] == key[0]) & (df_k['DATA'].dt.date == key[1])]

# %%
def get_coordinates_geopy(stop_name: str) -> list:
    """
        This function returns the coordinates of the stop using a dicionary that contains the coordinates of the stops. If the stop is not found in the dictionary, it uses geopy to find the coordinates.
        :param stop_name: the name of the stop
        :return: the coordinates of the stop
    """
    # Load data from the 'stop_converted.json' file into a dictionary
    with open('stop_converted.json') as f:
        data_aux = json.load(f)
        # Convert keys to lowercase and replace '-' with ' ' in the dictionary
        data = {k.lower().replace('-', ' '): v for k, v in data_aux.items()}

    # Standardize stop_name by replacing '-', apostrophes, and accents
    stop_name = stop_name.replace('-', ' ')
    stop_name = stop_name.replace('\'', '')
    stop_name = stop_name.replace('à', 'a')
    stop_name = stop_name.replace('è', 'e')
    stop_name = stop_name.replace('é', 'e')
    stop_name = stop_name.replace('ì', 'i')
    stop_name = stop_name.replace('ò', 'o')
    stop_name = stop_name.replace('ù', 'u')
    
    # Remove trailing space from stop_name
    if stop_name[-1] == ' ':
        stop_name = stop_name[:-1]

    # Check if stop_name or stop_name + ' ' exists in the data dictionary
    if stop_name.lower() in data.keys() or stop_name.lower() + ' ' in data.keys():
        # Found a matching stop_name
        stop_key = stop_name.lower() if stop_name.lower() in data.keys() else stop_name.lower() + ' '
        print(f'The stop {stop_name.lower()} is in the file stop_converted.json.')
        print(f'The coordinates of the stop {stop_name.lower()} are: {data[stop_key][1:]}')
        return [float(data[stop_key][1]), float(data[stop_key][2])]
    
    elif 'san' in stop_name.lower():
        # Check if the stop_name contains 'san', replace it with 's.', and try again
        stop_name = stop_name.lower().replace('san', 's.')
        if stop_name.lower() in data.keys() or stop_name.lower() + ' ' in data.keys():
            stop_key = stop_name.lower() if stop_name.lower() in data.keys() else stop_name.lower() + ' '
            print(f'The stop {stop_name.lower()} is in the file stop_converted.json.')
            print(f'The coordinates of the stop {stop_name.lower()} are: {data[stop_key][1:]}')
            return [float(data[stop_key][1]), float(data[stop_key][2])]
    
    elif len(stop_name.split(' ')) == 2:
        # Check if stop_name is composed of two words and try to find the first word
        stop_name_1 = stop_name.split(' ')[0].lower()
        if stop_name_1 in data.keys() or stop_name_1 + ' ' in data.keys():
            stop_key = stop_name_1 if stop_name_1 in data.keys() else stop_name_1 + ' '
            print(f'The stop {stop_name} is composed of two words. Consider only the first word {stop_name_1}.')
            print(f'The coordinates of the stop {stop_name_1} are: {data[stop_key][1:]}')
            return [float(data[stop_key][1]), float(data[stop_key][2])]

        # If the first word is not found, try to find the second word
        stop_name_2 = stop_name.split(' ')[1].lower()
        if stop_name_2 in data.keys() or stop_name_2 + ' ' in data.keys():
            stop_key = stop_name_2 if stop_name_2 in data.keys() else stop_name_2 + ' '
            print(f'The stop {stop_name} is composed of two words. Consider only the second word {stop_name_2}.')
            print(f'The coordinates of the stop {stop_name_2} are: {data[stop_key][1:]}')
            return [float(data[stop_key][1]), float(data[stop_key][2])]

    elif len(stop_name.split(' ')) == 3:
        # Check if stop_name is composed of three words and try to find the first two words
        stop_name_first_two = ' '.join(stop_name.split(' ')[:2]).lower()
        if stop_name_first_two in data.keys() or stop_name_first_two + ' ' in data.keys():
            stop_key = stop_name_first_two if stop_name_first_two in data.keys() else stop_name_first_two + ' '
            print(f'The stop {stop_name} is composed of three words. Consider only the first two words {stop_name_first_two}.')
            print(f'The coordinates of the stop {stop_name_first_two} are: {data[stop_key][1:]}')
            return [float(data[stop_key][1]), float(data[stop_key][2])]

    # If the stop is not found in the data dictionary, try to find it with geopy
    # print(f'The stop {stop_name} is not in the dataframe. Trying to find it with geopy.')
    # geolocator = Nominatim(user_agent="my-app")
    # location = geolocator.geocode(stop_name + ', Venezia')
    location = None
    
    if location is None:
        # Stop not found in geopy, return the coordinates of the center of Venice
        # print(f'The stop {stop_name} is not found in geopy.')
        # print(f'The coordinates of the stop {stop_name} are: [45.4371908, 12.3345898]')
        # return [45.4371908, 12.3345898]

        # Set the coordinates of the TERRA stop contained in the dictionary
        # print(f'The stop {stop_name} is not found: set the coordinates of the TERRA stop.')
        return get_coordinates_geopy('TERRA')

    elif (
        location.latitude < 44.0 or location.latitude > 46.0 or
        location.longitude < 11.0 or location.longitude > 13.0
    ):
        # Coordinates are not in Veneto, return the coordinates of the center of Venice
        print(f'The stop {stop_name} is not in Veneto.')
        print(f'The coordinates of the stop {stop_name} are: [45.4371908, 12.3345898]')
        return [45.4371908, 12.3345898]
    
    else:
        # Found coordinates using geopy
        print(f'The coordinates of the stop {stop_name} are: [{location.latitude}, {location.longitude}]')
        return [location.latitude, location.longitude]


# %%
def get_coordinates (df_c: pd.DataFrame) -> pd.DataFrame:
    """
        This function returns the coordinates using openstreetmap.
        :param df_c: the dataframe
        :return: the coordinates
    """
    # There aren't columns with the coordinates so you have to obtain the coordinates using openstreetmap
    # Create a new column with the coordinates
    df_c['COORDINATES'] = df_c['DESCRIZIONE'].apply(get_coordinates_geopy)
    # Split the coordinates in two columns
    df_c[['LATITUDE', 'LONGITUDE']] = pd.DataFrame(df_c['COORDINATES'].tolist(), index=df_c.index)
    # Drop the column 'COORDINATES'
    df_c.drop(columns=['COORDINATES'], inplace=True)
    # If coordinate are "-1": ["TERRA", 45.491853, 12.242548]} change the description to "TERRA (fantom stop)" 
    # df.loc[df['LATITUDE'] == 45.491853, 'DESCRIZIONE'] = 'TERRA (fantom stop)'
    return df_c

# %%
def plot_trajectory(df_c: pd.DataFrame, key: tuple, is_focus_on_ticket_code: bool = False, ticket_code: str = None, is_focus_on_tourists: bool = False, is_focus_on_workers: bool = False) -> None:
    """
        This function plots the trajectory of the user in the map.
        :param df: the dataframe
        :param key: the key
        :param is_focus_on_ticket_code: if True, the dictionary will be filtered by ticket code
        :param ticket_code: the ticket code to filter the dictionary
        :param is_focus_on_tourists: if True, the dictionary will be filtered by tourists
        :param is_focus_on_workers: if True, the dictionary will be filtered by workers
    """
    df_key = get_rows_from_key(df_c, key)
    # Get the coordinates of the trajectory
    df_key = get_coordinates(df_key)
    # Plot the trajectory
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.scatterplot(x='LONGITUDE', y='LATITUDE', data=df_key, hue='DESCRIZIONE', s=100, ax=ax)
    # Set the title
    if is_focus_on_ticket_code:
        ax.set_title('Trajectory of the user {} with ticket code {}'.format(key[0], ticket_code))
    elif is_focus_on_tourists:
        ax.set_title('Trajectory of the tourist {}'.format(key[0]))
    elif is_focus_on_workers:
        ax.set_title('Trajectory of the worker {}'.format(key[0]))
    else:
        ax.set_title('Trajectory of the user {}'.format(key[0]))
    # Set the legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    # Set the x and y labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Set the x and y ticks
    ax.set_xticks(np.arange(12.2, 12.5, 0.01))
    ax.set_yticks(np.arange(45.4, 45.6, 0.01))
    # Set the grid
    ax.grid(True)
    # Set the aspect ratio
    ax.set_aspect('equal', 'box')   

# %%
def represent_trajectory_on_map (df_c: pd.DataFrame, key: tuple) -> None:
    """
        This function represents the trajectory of the user in the map.
        :param df: the dataframe
        :param key: the key in the format (serial, day) where day can be None, that identifies the user
        :return: the map
    """

    df_key = get_rows_from_key(df_c, key)
    # Get the coordinates of the trajectory using openstreetmap
    df_key = get_coordinates(df_key)


    # print(df_key.head(50))
    
    # Create a map
    m = folium.Map(location=[45.437190, 12.334590], zoom_start=13)
    
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers to the map only if there isn't already a marker in the same position
    # First stop is green, last stop is red, other stops are blue
    # Insert first to all the first and last stops
    # If the first stop is the same of the last stop, insert only the first stop with black color

    # Define a list of stops already inserted
    stops_inserted = []
   
    if df_key.iloc[0]['LATITUDE'] == df_key.iloc[-1]['LATITUDE'] and df_key.iloc[0]['LONGITUDE'] == df_key.iloc[-1]['LONGITUDE']:
        print('The first stop is the same of the last stop.')
        folium.Marker(location=[df_key.iloc[0]['LATITUDE'], df_key.iloc[0]['LONGITUDE']], popup=df_key.iloc[0]['DESCRIZIONE'], icon=folium.Icon(color='black')).add_to(marker_cluster)
    else:
        folium.Marker(location=[df_key.iloc[0]['LATITUDE'], df_key.iloc[0]['LONGITUDE']], popup=df_key.iloc[0]['DESCRIZIONE'], icon=folium.Icon(color='green')).add_to(marker_cluster)
        # If the first stop is the same of the last stop, insert only the first stop
        folium.Marker(location=[df_key.iloc[-1]['LATITUDE'], df_key.iloc[-1]['LONGITUDE']], popup=df_key.iloc[-1]['DESCRIZIONE'], icon=folium.Icon(color='red')).add_to(marker_cluster)
    
    # Add the first and the last stop to the list of stops already inserted
    stops_inserted.append((df_key.iloc[0]['LATITUDE'], df_key.iloc[0]['LONGITUDE']))
    # If the first stop is the same of the last stop, insert only the first stop
    if df_key.iloc[0]['LATITUDE'] != df_key.iloc[-1]['LATITUDE'] and df_key.iloc[0]['LONGITUDE'] != df_key.iloc[-1]['LONGITUDE']:
        stops_inserted.append((df_key.iloc[-1]['LATITUDE'], df_key.iloc[-1]['LONGITUDE']))
        

    # Insert the other stops
    for i in range(1, len(df_key)-1):
        # If the stop is not already inserted, insert it
        if (df_key.iloc[i]['LATITUDE'], df_key.iloc[i]['LONGITUDE']) not in stops_inserted:
            stops_inserted.append((df_key.iloc[i]['LATITUDE'], df_key.iloc[i]['LONGITUDE']))
            folium.Marker(location=[df_key.iloc[i]['LATITUDE'], df_key.iloc[i]['LONGITUDE']], popup=df_key.iloc[i]['DESCRIZIONE']).add_to(marker_cluster)        

    # Highlight the trajectory of the user
    folium.PolyLine(locations=df_key[['LATITUDE', 'LONGITUDE']].values.tolist(), weight=2, color='red').add_to(m)

    # Insert the legend with the title Legend in bold
    legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 300px; height: 120px;
                    border:2px solid grey; z-index:9999; font-size:14px;
                    ">&nbsp; <b><u>Legend</u></b> <br>
                        &nbsp; First stop &nbsp; <i class="fa fa-map-marker fa-2x" style="color:green"></i><br>
                        &nbsp; Last stop &nbsp; <i class="fa fa-map-marker fa-2x" style="color:red"></i><br>
                        &nbsp; Coincident first and last stop &nbsp; <i class="fa fa-map-marker fa-2x" style="color:black"></i><br> 
        </div>
        '''
    m.get_root().html.add_child(folium.Element(legend_html))
    

    # Display the map
    display(m)

    # Save the map (if the folder doesn't exist, create it)
    import os
    if not os.path.exists('data/maps/' + subfolder):
        os.makedirs('data/maps/' + subfolder)
    m.save('data/maps/' + subfolder + '/map_' + file_name + '_key:' + str(key) + '.html')

# %%
def represent_trajectories_on_map_users (df_c: pd.DataFrame, list_keys:list) -> None:
    """
        This function represents the trajectories of the users in the map.
        :param df: the dataframe
        :param list_keys: the list of keys in the format (serial, day) where day can be None, that identifies the users
    """

    # Get a list of colors
    colors = ['red', 'blue', 'green', 'black', 'orange', 'purple', 'darkred', 'lightred', #'beige'
                 'darkblue', 
              'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 
              'lightgray', 'blue', 'green', 'red', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 
              'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

    # Create a map
    m = folium.Map(location=[45.437190, 12.334590], zoom_start=13)

    # Create a marker cluster
    marker_cluster = MarkerCluster().add_to(m)

    # Define a list of stops already inserted
    # Each element of the list is a tuple (latitude, longitude)
    stops_inserted = []
    
    j = 0

    # For each key in the list of keys
    for key in list_keys:
        if j == len(colors):
            j = 0
        # Extract the values of the key
        key = extract_values_key(key)
        df_key = get_rows_from_key(df_c, key)
        # Get the coordinates of the trajectory
        df_key = get_coordinates(df_key)

        # Add markers to the map only if there isn't already a marker in the same position
        for i in range(0, len(df_key)):
            # If the stop is not already inserted, insert it
            if (df_key.iloc[i]['LATITUDE'], df_key.iloc[i]['LONGITUDE']) not in stops_inserted:
                stops_inserted.append((df_key.iloc[i]['LATITUDE'], df_key.iloc[i]['LONGITUDE']))
                folium.Marker(location=[df_key.iloc[i]['LATITUDE'], df_key.iloc[i]['LONGITUDE']], popup=df_key.iloc[i]['DESCRIZIONE']).add_to(marker_cluster)

        # Highlight the trajectory of the user
        folium.PolyLine(locations=df_key[['LATITUDE', 'LONGITUDE']].values.tolist(), weight=2, color=colors[j]).add_to(m)


        j += 1

    # Display the map
    display(m)

    # Save the map (if the folder doesn't exist, create it)
    # import os
    # if not os.path.exists('data/maps/' + subfolder):
    #     os.makedirs('data/maps/' + subfolder)
    # m.save('data/maps/' + subfolder + '/map_' + file_name + '_users.html')

# %%
# Given the dataset, get the geolocation of the stops and add them to the dataset
df = get_coordinates(df)

# %%
df.head()

# %%
# Export the dataframe to a csv file
date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

path = 'data/processed/' + subfolder + '/df_' + file_name + '_' + note + '_' + date + '_GEO.csv'
df.to_csv(path, index=False)

print("The dataset has been exported to a csv file in the folder " + path)


