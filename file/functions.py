import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def haversine_distance(coord1, coord2):
    """
        Calculate the distance between two points on Earth using the haversine formula.
        The haversine formula determines the great-circle distance between two points on a sphere given their longitudes and latitudes.
        The haversin formula is specified as:
            a = sin²(Δlat/2) + cos(lat1).cos(lat2).sin²(Δlong/2)
            c = 2.atan2(√a, √(1−a))
            d = R.c
        where:
            lat1, long1 = Latitude and Longitude of point 1 (in decimal degrees)
            lat2, long2 = Latitude and Longitude of point 2 (in decimal degrees)
            R = Radius of the Earth in kilometers
            Δlat = lat2− lat1
            Δlong = long2− long1

        :param coord1: Tuple of (latitude, longitude) for point 1
        :param coord2: Tuple of (latitude, longitude) for point 2
        :return: Distance between the two coordinates in kilometers
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    R = 6371  # Radius of the Earth in kilometers
    
    # Convert decimal degrees to radians
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    # Apply haversine formula
    # a = sin²(Δlat/2) + cos(lat1).cos(lat2).sin²(Δlong/2)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2

    # c = 2.atan2(√a, √(1−a))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # d = R.c
    distance = R * c

    # print('Distance between stops: ', distance)
    return distance

def euclidean_distance(vector1, vector2):
    """
    Normalize the Euclidean distance between two vectors to the range [0, 1].

    :param vector1: First vector (numpy array).
    :param vector2: Second vector (numpy array), should have the same dimension as vector1.
    :return: Normalized Euclidean distance between the two vectors (value between 0 and 1).
    """

    # Ensure both vectors have the same dimension
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimension for Euclidean distance calculation.")

    # Calculate the Euclidean distance
    euclidean_distance = np.linalg.norm(vector1 - vector2)
    return euclidean_distance

def haversine_distance_normalized(coord1, coord2, min_distance, max_distance):
    distance = haversine_distance(coord1, coord2)
    normalized_distance = (distance - min_distance) / (max_distance - min_distance)
    #normalized_distance = 1-normalized_distance
    #print(f'normalized_distance: {normalized_distance}')
    return normalized_distance

def euclidean_distance_normalized(vector1, vector2, min_distance, max_distance):
    distance = euclidean_distance(vector1, vector2)
    normalized_distance = (distance - min_distance) / (max_distance - min_distance)
    #normalized_distance = 1-normalized_distance
    #print(f'normalized_distance: {normalized_distance}')
    
    return normalized_distance


def custom_distance(stop1, stop2, coord_weight, similarity_weight, min_distance_hd, max_distance_hd, min_distance_ed, max_distance_ed):
    """
        Calculate the custom distance between two stops.
        The custom distance is a weighted combination of the haversine distance between the two stops and the cosine similarity between the two stops.
        The custom distance is mathematically defined as:
            custom_distance = coord_weight * haversine_distance + similarity_weight * cosine_similarity_distance
        where:
            coord_weight = Weight for haversine distance
            similarity_weight = Weight for cosine similarity distance
        :param stop1: Tuple of (latitude, longitude, counts for each ticket code) for stop 1
        :param stop2: Tuple of (latitude, longitude, counts for each ticket code) for stop 2
        :param coord_weight: Weight for haversine distance
        :param similarity_weight: Weight for cosine similarity distance
        :return: Custom distance between the two stops
    """
    # Calculate distances
    # Calculate haversine distance between two stops
    coord_distance = haversine_distance_normalized((stop1[0], stop1[1]), (stop2[0], stop2[1]), min_distance_hd, max_distance_hd)
    # Calculate cosine similarity between two stops
    count_similarity = euclidean_distance_normalized(stop1[2:], stop2[2:], min_distance_ed, max_distance_ed)
    
    # Combine distances with appropriate weights
    combined_distance = coord_weight*coord_distance+similarity_weight*count_similarity
    #print(f'stop 1: {stop1[2]}, stop2: {stop2[2]}, combined_distance: {combined_distance}')
    return combined_distance


def find_csv_files(folder_path: str) -> list:
    """
        This function returns a list of all the csv files in the specified folder.
        :param folder_path: the path of the folder
        :return: a list of all the csv files in the specified folder
    """

    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # File extension is .txt or .csv
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    # Sort the list of txt files in alphabetical order
    csv_files.sort()

    return csv_files

def choose_dataset(txt_files: list) -> str:
    """
        This function returns the path of the txt file chosen by the user.
        :param txt_files: the list of txt files
        :return: the path of the txt file chosen by the user
    """
    if not txt_files:
        print("No TXT file found.")
        return "None"
    if len(txt_files) == 1:
        print("The following file was found:")
    else:
        print("The following files were found:")
    for i, file_path in enumerate(txt_files):
        print(f"{i+1}. {file_path}")
    while True:
        choice = input("Enter the number corresponding to the dataset you wish to use (0 to exit): ")
        if not choice.isdigit():
            print("Enter a valid number.")
            continue
        choice = int(choice)
        if choice == 0:
            return "None"
        if choice < 1 or choice > len(txt_files):
            print("Enter a valid number.")
            continue
        return txt_files[choice - 1]
    