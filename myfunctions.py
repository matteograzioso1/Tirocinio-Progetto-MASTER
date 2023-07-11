import pandas as pd
import json
import os

def find_txt_files(folder_path: str) -> list:
    """
        This function returns a list of all the txt files in the specified folder.
        :param folder_path: the path of the folder
        :return: a list of all the txt files in the specified folder
    """

    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # File extension is .txt or .csv
            if file.endswith(".txt") or file.endswith(".csv"):
                txt_files.append(os.path.join(root, file))
    # Sort the list of txt files in alphabetical order
    txt_files.sort()

    return txt_files

def choose_dataset(txt_files: list) -> str:
    """
        This function returns the path of the txt file chosen by the user.
        :param txt_files: the list of txt files
        :return: the path of the txt file chosen by the user
    """
    if not txt_files:
        print("No TXT file found.")
        return "None"
    print("The following TXT files were found:")
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
    

def check_folder_exists(path: str) -> bool:
    """
        This function checks if a folder exists and creates it if it doesn't exist.
        :param path: the path of the folder
        :return: True if the folder exists, False otherwise
    """
    import os
    if os.path.exists(path):
        print("The folder already exists.")
        return True
    else:
        # Create the folder
        os.mkdir(path)
        print("The folder has been created.")
        return False
    

def get_ticket_code_description(ticket_code: str) -> str:
    """
      Given the key of the ticket code, return the description of the ticket code contained in the dictionary dict_ticket_code.json
      Input:
        ticket_code - a string that is the key of the dictionary
      Output:
        description - a string that is the description of the ticket code, value of the dictionary
    """
    with open('data/dictionaries/dict_ticket_codes.json') as f:
        data = json.load(f)
    description = data[ticket_code]
    return description


def focus_on_ticket_code(df_tc: pd.DataFrame, ticket_code: str) -> pd.DataFrame:
    """
        This function returns a dataframe with only the rows of the specified ticket code.
        :param df: the dataframe
        :param ticket_code: the ticket code
        :return: the dataframe with only the rows of the specified ticket code
    """
    # Convert the column ticket_code to string
    df_tc['TICKET_CODE'] = df_tc['TICKET_CODE'].astype(str)

    # Select only the rows of the specified ticket code
    df_tc = df_tc[df_tc['TICKET_CODE'] == ticket_code]
    return df_tc

def focus_on_ticket_code_list(df_tcl: pd.DataFrame, ticket_code_list: list) -> pd.DataFrame:
    """
        This function returns a dataframe with only the rows of the specified ticket code list.
        :param df: the dataframe
        :param ticket_code_list: the ticket code list
        :return: the dataframe with only the rows of the specified ticket code list
    """
    # Convert the column ticket_code to string
    df_tcl['TICKET_CODE'] = df_tcl['TICKET_CODE'].astype(str)

    # Select only the rows of the specified ticket code
    df_tcl = df_tcl[df_tcl['TICKET_CODE'].isin(ticket_code_list)]
    return df_tcl

def focus_on_ticket_type(df_tt: pd.DataFrame, ticket_type: str) -> pd.DataFrame:
    """
        This function returns a dataframe with only the rows of the specified ticket type.
        :param df: the dataframe
        :param ticket_type: the ticket type
        :return: the dataframe with only the rows of the specified ticket type
    """
    # Select only the rows of the specified ticket type
    df_tt = df_tt[df_tt['DESCRIZIONE_TITOLO'] == ticket_type]
    df_tt.head()
    return df_tt

def focus_on_period(df_p: pd.DataFrame, start_date: str, end_date: str):
    """
        This function focuses on the specified period.
        :param df: the dataframe
        :param start_date: the start date of the period
        :param end_date: the end date of the period
        :return: the dataframe focused on the specified period
    """
    return df_p[(df_p['DATA'] >= start_date) & (df_p['DATA'] <= end_date)]

def open_dict_ticket_codes () -> dict:
    """
        This function returns the dictionary of ticket codes.
        :return: the dictionary of ticket codes
    """
    with open('data/dictionaries/dict_ticket_codes.json') as f:
        data = json.load(f)
    return data