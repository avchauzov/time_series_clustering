# Time Series Clustering and Merging

## Description
This project provides a set of Python functions for time series clustering and merging. It includes capabilities for flattening nested lists, computing distance matrices using Dynamic Time Warping (DTW) and Spearman correlation, and merging time series based on these computed distances or correlations.

## Notes
1. **Example Usage**: For comprehensive examples and usage scenarios, refer to the [test.ipynb](https://github.com/avchauzov/time_series_clustering/blob/master/test.ipynb) notebook included in this repository.
2. **Tunability and Adaptability**:
- _Scaling Variations_: The scaling method can be tailored to a more robust variant, accommodating a wider range of datasets.
- _Compression of Complex Series_: In cases involving long and complex time series, a compression/simplification approach may be preferable. This strategy has been effectively utilized in production settings, yielding notable results. 
- _Flexible Aggregation Methods_: The aggregation methods employed in the clustering process are versatile and can be modified to meet diverse requirements.
3. **Distance Matrix Computation**: The distance matrix computation is pivotal. It is crucial to use the DTW-C-based version for enhanced efficiency and accuracy.
4. **Clustering Stopping Rule**: The criteria for terminating the clustering process can be improved. Specifically, tailoring it for the Spearman method may enhance the outcome.
5. **Additional Applications**: This methodology has been successfully applied for clustering textual entities using their BERT embeddings with the cosine metric. This application has shown superior performance compared to other clustering techniques.

## Installation
To use these functions, clone the repository and import the functions in your Python script.
```
git clone https://github.com/avchauzov/time_series_clustering
from functions import *
```

## Usage
Here's a basic example of how to use these functions:
```python
import numpy as np
from functions import compute_spearman_distance_matrix, merge_time_series, flatten_indices


# Define the length of each time series and the number of series to generate
n = 4  # Length of each time series
num_series = 32  # Number of time series to generate
number_of_clusters = 16  # Desired number of clusters for merging time series

# Generate random time series data with a specified number of series and length
time_series_data = np.random.randn(num_series, n)

# Compute the Spearman distance matrix for the time series data
# 'time_series_indices' will hold the indices of the time series in the original dataset
distance_matrix, time_series_indices = compute_spearman_distance_matrix(time_series_data)

# Merge the time series based on the computed distance matrix into a specified number of clusters
# 'aggregation' method is set to 'mean' for merging
time_series_indices_result = merge_time_series(distance_matrix, time_series_indices, number_of_clusters, aggregation='mean')

# Flatten the hierarchical indices obtained after merging to get a simple list of indices
time_series_indices_result = flatten_indices(time_series_indices_result)

# Print the final flattened indices of the merged time series
print(time_series_indices_result)
```

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.

## Contact
For any questions or suggestions, please feel free to contact [avchauzov@gmail.com](mailto:avchauzov@gmail.com).
