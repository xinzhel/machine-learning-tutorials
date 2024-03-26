import numpy as np
import matplotlib.pyplot as plt

def generate_sample(sample_size, pdf_or_pmf=np.random.uniform):
    return [pdf_or_pmf() for _ in range(sample_size)]