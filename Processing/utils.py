import pandas as pd
import numpy as np

def parse_embedding_flexible(s):
    if isinstance(s, str):
        s = s.replace('[', '').replace(']', '').replace('\n', ' ')
        # Replace commas with spaces
        s = s.replace(',', ' ')
        return np.array([float(x) for x in s.split() if x.strip() != ''])
    elif isinstance(s, (list, np.ndarray)):
        return np.array(s)
    else:
        return np.nan