import pandas as pd
import json

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