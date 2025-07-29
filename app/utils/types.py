import numpy as np

Ragged = np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
RaggedMemmap = np.memmap | tuple[np.memmap, np.memmap, np.memmap]
