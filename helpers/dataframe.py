import pandas as pd
import math
from itertools import product


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


def generate_fetch_objects(base_url: str, subjects: list[str]) -> list[dict]:
    """
    Generates a list of fetch objects that can be used to fetch the data from the server.
    """
    signals = ["HR.csv", "ACC.csv", "BVP.csv", "EDA.csv", "TEMP.csv"]
    fetch_objects = [{ "url": "/".join([base_url] + list(i)), "subject": i[0], "signal": i[1] } for i in product(subjects, signals)]

    return fetch_objects


def concatenate_dataframe_from_fetch_objects(fetch_objects: list[dict]) -> tuple[pd.DataFrame]:
    """
    Concatenates all the dataframes from the given fetch objects into one dataframe.
    """
    signal_columns = {
        "HR.csv": ["HR"],
        "ACC.csv": ["X", "Y", "Z"],
        "BVP.csv": ["BVP"],
        "EDA.csv": ["EDA"],
        "TEMP.csv": ["TEMP"],
    }

    acc_df = pd.DataFrame(columns=signal_columns['ACC.csv'])
    hr_df = pd.DataFrame(columns=signal_columns['HR.csv'])
    eda_df = pd.DataFrame(columns=signal_columns['EDA.csv'])
    temp_df = pd.DataFrame(columns=signal_columns['TEMP.csv'])
    bvp_df = pd.DataFrame(columns=signal_columns['BVP.csv'])

    for i in fetch_objects:
        print(f"Fetching {i['signal']} for subject {i['subject']}")
        raw = pd.read_csv(i["url"], header=None)
        column_names = signal_columns[i["signal"]]

        # Concat each dataframe with the one from the previous iteration
        if i['signal'] == 'ACC.csv':
            acc_df = pd.concat([acc_df, convert_from_raw(raw, column_names, int(i['subject'][1:]))])
        elif i['signal'] == 'HR.csv':
            hr_df = pd.concat([hr_df, convert_from_raw(raw, column_names, int(i['subject'][1:]))])
        elif i['signal'] == 'EDA.csv':
            eda_df = pd.concat([eda_df, convert_from_raw(raw, column_names, int(i['subject'][1:]))])
        elif i['signal'] == 'TEMP.csv':
            temp_df = pd.concat([temp_df, convert_from_raw(raw, column_names, int(i['subject'][1:]))])
        elif i['signal'] == 'BVP.csv':
            bvp_df = pd.concat([bvp_df, convert_from_raw(raw, column_names, int(i['subject'][1:]))])

    return acc_df, hr_df, eda_df, temp_df, bvp_df


def combine_dataframe(acc_df: pd.DataFrame, hr_df: pd.DataFrame, eda_df: pd.DataFrame, temp_df: pd.DataFrame, bvp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines all the dataframes into one dataframe.
    """
    df = pd.merge(acc_df, hr_df, on=['Timestamp', 'Subject'], how='outer')
    df = pd.merge(df, eda_df, on=['Timestamp', 'Subject'], how='outer')
    df = pd.merge(df, temp_df, on=['Timestamp', 'Subject'], how='outer')
    df = pd.merge(df, bvp_df, on=['Timestamp', 'Subject'], how='outer')

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills the missing values in the dataframe with the mean of the previous and next values.
    """
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    return df


def generate_tag_objects(base_url: str, subjects: list[str]) -> list[dict]:
    """
    Fetches the tag objects from the server.
    """
    tags = [f"tags_{i}.csv" for i in subjects]

    tag_objects = [{ "url": "/".join([base_url] + list(i)), "subject": i[0], "tag": i[1] } for i in zip(subjects, tags)]

    return tag_objects


def fetch_timestamps_from_tag_objects(tag_objects: list[dict]) -> list[dict]:
    """
    Fetches the timestamps from the tag objects.
    """
    timestamps = []

    for tag_object in tag_objects:
        df = pd.read_csv(tag_object['url'], header=None)
        # Here we assume that the first six timestamps in the tags file are useful.
        # It's not fully clear from the paper, but since there are 3 stress-inducing tasks for a subject,
        # only six timestamps should matter (start + end time).
        s = df.iloc[:6]
        timestamps.extend([round(i) for i in s[0].to_list()])

    return timestamps


def label_dataframe_from_timestamps(df: pd.DataFrame, timestamps: list[dict]) -> pd.DataFrame:
    """
    Labels the dataframe with the given timestamps.
    """

    prev_index = 0

    # Iterate over the timestamps and change the label column
    for idx, i in enumerate(timestamps):
        # Check where the timestamp is present in the dataset
        index = int(df[df['Timestamp'] == float(i * 1000)].index[0])

        # If it's the first timestamp, we assume it's the start of a stressful task.
        # Logically, the next timestamp would be the end of the task,
        # which would mean the next timestamps would not involve a stressful task.
        # Then the next timestamp would be the start of the next task and so on.
        # In this way we can label the dataset appropriately based on the above assumptions.
        if (idx + 1) % 2 == 0:
            # Label = 1 means that the subject is under stress.
            df.loc[prev_index:index, 'Label'] = 1 
        else:
            df.loc[prev_index:index, 'Label'] = 0
        prev_index = index

    # Label the rest of the rows
    df.loc[prev_index:, 'Label'] = 0

    return df