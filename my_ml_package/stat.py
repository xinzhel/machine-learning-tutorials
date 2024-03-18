
import numpy as np

def generate_sample(sample_size, pdf_or_pmf=np.random.uniform):
    return [pdf_or_pmf() for _ in range(sample_size)]

def find_position_for_percentile(percentile, sorted_data):
    """ Returns the position of a given percentile in a sorted list of data.
    Percentile: A value below which a given percentage of data falls.
    """
    n = len(sorted_data)
    position = (percentile/100) * (n + 1)
    return position


def find_percentile(percentile, sorted_data):
    """ Returns the value of a given percentile in a sorted list of data.
    Percentile: A value below which a given percentage of data falls.
    """
    position = find_position_for_percentile(percentile, sorted_data)
    if position.is_integer():
        return sorted_data[int(position) - 1]
    else:
        k = int(position) 
        fraction = position - k
        return sorted_data[k - 1] + fraction * (sorted_data[k] - sorted_data[k - 1]) # Linear interpolation

# Interquartile Range (IQR)
def find_median(sorted_data):
    n = len(sorted_data)
    if n % 2 == 0:
        return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    else:
        return sorted_data[n//2]

def calculate_quartiles(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Calculate Q1
    lower_half = sorted_data[:n//2]
    Q1 = find_median(lower_half)
    
    # calculate Q2
    Q2 = find_median(sorted_data)

    # Calculate Q3
    if n % 2 == 0:
        upper_half = sorted_data[n//2:]
    else: # len(data)%2 == 1
        upper_half = sorted_data[n//2+1:]
    Q3 = find_median(upper_half)
    
    return Q1, Q2, Q3

def calculate_iqr_bounds(data):
    Q1, _, Q3 = calculate_quartiles(data)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound
