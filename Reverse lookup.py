def reverse_lookup_binary(d):
    """
    Perform a reverse lookup on a dictionary whose values are binary (0 or 1).

    The function returns a dictionary mapping each value in the input dictionary
    to a list of keys that map to that value.

    Parameters
    ----------
    d : dict
        A dictionary whose values are expected to be 0 or 1.

    Returns
    -------
    dict
        A dictionary where each key is a value from d (0 or 1) and each value
        is a list of keys from d that mapped to it.
    """
    assert isinstance(d, dict), "Input must be a dictionary."

    result = {}

    for key, value in d.items():
        assert value in (0, 1), "Dictionary values must be 0 or 1."

        if value not in result:
            result[value] = []

        result[value].append(key)

    return result
