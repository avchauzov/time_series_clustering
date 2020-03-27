# %%

from copy import deepcopy

import numpy as np
import pandas as pd
from fastdtw import fastdtw

# %%

dataDaily = pd.read_csv('data/1.dataDaily.csv', index_col = 0)

# %%

dataDaily.head(25)

# %%

THRESHOLD = 1000

trajectoriesSet = {}

for column in list(dataDaily):
	
	if column == 'index':
		continue
	
	tempArray = dataDaily[column].values
	
	indexFirst = 0
	for index in range(len(tempArray)):
		
		if tempArray[index] != 0:
			indexFirst = index
			break
	
	tempArray = tempArray[index:]
	
	indexLast = len(tempArray)
	for index in range(len(tempArray) - 1, 0, -1):
		
		if tempArray[index] != 0:
			indexLast = index
			break
	
	tempArray = tempArray[: indexLast + 1]
	
	print(tempArray[0], tempArray[-1])
	
	trajectoriesSet[(column,)] = [list(tempArray)]
	
	if len(trajectoriesSet.keys()) >= 5:
		break

# %%

trajectories = deepcopy(trajectoriesSet)
distanceMatrixDictionary = {}

iteration = 1
while True:
	distanceMatrix = np.empty((len(trajectories), len(trajectories),))
	distanceMatrix[:] = np.nan
	
	for index1, (filter1, trajectory1) in enumerate(trajectories.items()):
		tempArray = []
		
		for index2, (filter2, trajectory2) in enumerate(trajectories.items()):
			
			if index1 > index2:
				continue
			
			elif index1 == index2:
				continue
			
			else:
				unionFilter = filter1 + filter2
				sorted(unionFilter)
				
				if unionFilter in distanceMatrixDictionary.keys():
					distanceMatrix[index1][index2] = distanceMatrixDictionary.get(unionFilter)
					
					continue
				
				metric = []
				for subItem1 in trajectory1:
					
					for subItem2 in trajectory2:
						# metric.append(dtw.distance_fast(subItem1, subItem2, psi=1))
						# print(fastdtw(subItem1, subItem2)[0])
						metric.append(fastdtw(subItem1, subItem2)[0])
				
				metric = max(metric)
				
				distanceMatrix[index1][index2] = metric
				distanceMatrixDictionary[unionFilter] = metric
	
	minValue = np.min(list(distanceMatrixDictionary.values()))
	
	if minValue > THRESHOLD:
		print(minValue, THRESHOLD)
		break
	
	minIndices = np.where(distanceMatrix == minValue)
	minIndices = list(zip(minIndices[0], minIndices[1]))
	
	minIndex = minIndices[0]
	
	filter1 = list(trajectories.keys())[minIndex[0]]
	filter2 = list(trajectories.keys())[minIndex[1]]
	
	trajectory1 = trajectories.get(filter1)
	trajectory2 = trajectories.get(filter2)
	
	unionFilter = filter1 + filter2
	sorted(unionFilter)
	
	trajectoryGroup = trajectory1 + trajectory2
	
	trajectories = {key: value for key, value in trajectories.items()
	                if all(value not in unionFilter for value in key)}
	
	distanceMatrixDictionary = {key: value for key, value in distanceMatrixDictionary.items()
	                            if all(value not in unionFilter for value in key)}
	
	trajectories[unionFilter] = trajectoryGroup
	
	print(iteration, 'finished!')
	iteration += 1
	
	if len(list(trajectories.keys())) == 1:
		break

# %%

for key, value in trajectories.items():
	print(key, len(value))
