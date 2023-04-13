def generate_fetch_objects(base_url: str, subject_nums: list[int], signals: list[str]) -> list[dict]:
    """
    Generates a list of fetch objects that can be used to fetch the data from the server.
    """
    fetch_objects = []

    for subject_num in subject_nums:
        for signal in signals:
            fetch_objects.append({
                'url': f'{base_url}/{subject_num}/{signal}',
                'subject_num': subject_num,
                'signal': signal
            })

    return fetch_objects

