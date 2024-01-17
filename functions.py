"""
Time Series Clustering and Merging Script

This script provides functions for flattening nested lists, computing distance matrices
using Dynamic Time Warping (DTW) and Spearman correlation, and merging time series based
on DTW distances or Spearman correlations.

The main functions provided in this script are:
- flatten_nested_list: Flatten a nested list into a flat list.
- flatten_indices: Flatten a list of hierarchical indices.
- compute_dtw_distance_matrix: Compute the DTW distance matrix between time series.
- compute_spearman_distance_matrix: Compute the Spearman correlation-based distance matrix.
- merge_time_series: Merge time series based on distance metrics.
"""

import numpy as np
from dtaidistance import dtw
from scipy.stats import spearmanr
from tqdm import tqdm


def flatten_nested_list(nested_list):
	"""
	Flatten a nested list, recursively combining all elements into a single flat list.

	Parameters:
		nested_list (list): The nested list to flatten.

	Returns:
		list: A flat list containing all elements from the nested list.
	"""
	flat_list = []
	for item in nested_list:
		if isinstance(item, list):
			# If the item is a list, recursively flatten it
			flat_list.extend(flatten_nested_list(item))
		else:
			# If the item is not a list, append it to the flat_list
			flat_list.append(item)
	return flat_list


def flatten_indices(time_series_indices):
	"""
	Flatten a list of hierarchical indices into a single flat list.

	Parameters:
		time_series_indices (list): The list of hierarchical indices to flatten.

	Returns:
		list: A flat list containing all indices from the hierarchical structure.
	"""
	flattened_indices = []
	
	for item in time_series_indices:
		if isinstance(item, int):
			# If the item is an integer, wrap it in a list and append
			flattened_indices.append([item])
		else:
			# If the item is a nested list, flatten it and append the sorted result
			item = flatten_nested_list(item)
			flattened_indices.append(sorted(item))
	
	return flattened_indices


def compute_dtw_distance_matrix(time_series_data):
	"""
	Compute the Dynamic Time Warping (DTW) distance matrix between time series.

	Parameters:
		time_series_data (ndarray): 2D NumPy array where each row represents a time series.

	Returns:
		ndarray: The DTW distance matrix.
		list: A list of time series indices.
	"""
	num_series = time_series_data.shape[0]
	
	# Initialize time series indices and distance matrix
	time_series_indices = list(range(num_series))
	distance_matrix = dtw.distance_matrix_fast(time_series_data)
	np.fill_diagonal(distance_matrix, np.nan)
	
	return distance_matrix, time_series_indices


def compute_spearman_distance_matrix(time_series_data):
	"""
	Compute the Spearman correlation-based distance matrix between time series.

	Parameters:
		time_series_data (ndarray): 2D NumPy array where each row represents a time series.

	Returns:
		ndarray: The Spearman correlation-based distance matrix.
		list: A list of time series indices.
	"""
	num_series = time_series_data.shape[0]
	
	# Initialize time series indices and distance matrix
	time_series_indices = list(range(num_series))
	
	# Calculate the Spearman correlation matrix
	spearman_corr_matrix, _ = spearmanr(time_series_data, axis=1)
	
	# Convert Spearman correlation matrix to a distance matrix
	distance_matrix = 1 - spearman_corr_matrix
	np.fill_diagonal(distance_matrix, np.nan)
	
	return distance_matrix, time_series_indices


def merge_time_series(distance_matrix, time_series_indices, number_of_clusters, aggregation='min'):
	"""
	Merge time series based on dynamic time warping distances until a specified number of clusters is reached.

	Parameters:
		distance_matrix (ndarray): The distance matrix between time series.
		time_series_indices (list): A list of time series indices.
		number_of_clusters (int): The desired number of clusters.
		aggregation (str): The aggregation method ('min' or 'mean') for merging.

	Returns:
		list: A list of hierarchical indices representing merged time series.
	"""
	with tqdm(total=number_of_clusters, desc='Merging Process') as pbar:
		while distance_matrix.shape[0] != number_of_clusters:
			min_value_index = np.nanargmin(distance_matrix)
			
			# Convert the 1D index to row and column indices
			min_row, min_column = np.unravel_index(min_value_index, distance_matrix.shape)
			
			if aggregation == 'min':
				new_row_column = np.nanmin([distance_matrix[min_row, :], distance_matrix[min_column, :]], axis=0)
			else:
				new_row_column = np.nanmean([distance_matrix[min_row, :], distance_matrix[min_column, :]], axis=0)
			
			# Insert the new row and column at the end
			distance_matrix = np.insert(distance_matrix, distance_matrix.shape[0], new_row_column, axis=0)
			
			# Append np.nan to the new row and column
			new_row_column = np.append(new_row_column, np.nan)
			
			distance_matrix = np.insert(distance_matrix, distance_matrix.shape[1], new_row_column, axis=1)
			
			# Update time_series_indices
			time_series_indices.append([time_series_indices[min_row], time_series_indices[min_column]])
			
			# Delete rows and columns
			distance_matrix = np.delete(distance_matrix, [min_row, min_column], axis=0)
			distance_matrix = np.delete(distance_matrix, [min_row, min_column], axis=1)
			
			# Remove indices from time_series_indices
			time_series_indices = [value for index, value in enumerate(time_series_indices) if index not in [min_row, min_column]]
			
			pbar.update()  # 1; Update the progress bar
	
	return time_series_indices
