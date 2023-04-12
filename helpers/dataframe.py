import pandas as pd
import math

def extract_timestamp_rate(df: pd.DataFrame):
    """
    Returns the starting timestamp and sampling rate of the dataset.
    The first row contains the timestamp and the second row contains the sample rate.
    The actual data is then in the rows afterwards.

    For some signals, timestamp and rate could be repeating into multiple columns,
    hence we only need to return one of the items of the extracted lists
    """
    start_ts = df.iloc[0, 0]
    rate = df.iloc[1, 0]

    if type(start_ts) is list:
        start_ts = start_ts[0]
    if type(rate) is list:
        rate = rate[0]

    return (start_ts, rate)


def generate_timestamps(start: float, rate: float, length: int):
    """
    Returns a list of all timestamps after the start time given the rate and length of the list

    Note: Unix time is multiplied by 1000 to support miliseconds. This is due to most of the signal rates being greater than 1.
    """
    return [math.floor((start + i/rate) * 1000) for i in range(length)]



def convert_from_raw(raw: pd.DataFrame, column_names: list[str], subject_num: int) -> pd.DataFrame:
    """
    Processes the given raw dataset and converts it into a dataset with timestamps mapped to the provided column names.
    """
    start, rate = extract_timestamp_rate(raw)

    df = pd.DataFrame(raw.iloc[2:].values, columns=column_names)

    timestamps = generate_timestamps(start, rate, len(raw) - 2)
    df['Timestamp'] = timestamps
    
    df['Subject'] = pd.Series([subject_num]*len(df), dtype=int)

    return df