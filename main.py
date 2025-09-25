import hashlib

def read_input_file(file_name):
    """
    Read the entire contents of a text file.
    Parameters
    ----------
    file_name : str
        Path to the text file to read.

    Returns
    -------
    str
        The full contents of the file as a single string.
    """
    with open(file_name, "r", encoding="utf-8") as file:
        content = file.read()
    return content

def w_shingles(text, w, lam):
    """
    Generate (w, λ)-shingles from input text.
    This function splits the text into tokens (by whitespace)
    and then slides a window of size `w` over the tokens,
    stepping by `lam` each time to create shingles.

    Parameters
    ----------
    text : str
        The input text to shingle.
    w : int
        The window size (number of tokens per shingle).
    lam : int
        The step size between consecutive shingles.

    Returns
    -------
    list of tuple
        Each tuple contains `w` consecutive tokens.
    """
    # Tokenize based on spaces
    tokens = text.split()
    shingles = []

    # Slide window of size w across tokens with step lam
    for i in range(0, len(tokens) - w + 1, lam):
        shingle = tuple(tokens[i:i + w])  # make tuple
        shingles.append(shingle)

    return shingles


def hash_shingles(shingles):
    """
    Convert shingles to MD5 hashes.
    Each shingle is assumed to be a tuple of tokens (strings).
    The function joins the tokens with a separator, encodes to bytes,
    and computes an MD5 hash for each shingle.

    Parameters
    ----------
    shingles : list of tuple
        List of shingles, where each shingle is a tuple of tokens.

    Returns
    -------
    list of str
        List of MD5 hash strings corresponding to each shingle.
    """
    hashes = []
    for shingle in shingles:
        # Join tokens using a separator; you can lowercase here if desired
        shingle_str = "|".join(shingle)  # skip .lower() if you care about case
        # Encode to bytes and hash using MD5
        shingle_hash = hashlib.md5(shingle_str.encode("utf-8")).hexdigest()
        hashes.append(shingle_hash)

    return hashes


def union(list1, list2):
    """
    Compute the union of two lists.

    Parameters
    ----------
    list1 : list
    list2 : list

    Returns
    -------
    list
        Elements present in either list1 or list2 (unique).
    """
    return list(set(list1) | set(list2))


def intersection(list1, list2):
    """
    Compute the intersection of two lists.

    Parameters
    ----------
    list1 : list
    list2 : list

    Returns
    -------
    list
        Elements present in both list1 and list2 (unique).
    """
    return list(set(list1) & set(list2))


def main():
    # 1. Read input file
    ip_data = read_input_file("Athens_Georgia.txt")
    print("Input read test:\n", ip_data)

    # 2. Generate (w, λ)-shingles
    test_shingles = w_shingles(ip_data, w=3, lam=1)
    print("\nGenerated shingles:\n", test_shingles)

    # 3. Compute MD5 hashes of shingles
    hashed_shingles = hash_shingles(test_shingles)
    print("\nMD5 hashes of shingles:\n", hashed_shingles)


if __name__ == "__main__":
    main()
