# %%


from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import spearmanr
from sklearn.preprocessing import robust_scale

# %%

data = pd.read_csv('data/1.data.csv', index_col = 0)

# %%

data.head(25)

# %%

data = data[list(data)[: 100]]

tempDictionary = {}
for column in list(data):
	tempArray = tuple([str(value) for value in data[column].values])
	
	if tempArray not in tempDictionary.keys():
		tempDictionary[tempArray] = [column]
	
	else:
		tempDictionary[tempArray].append(column)

for key, value in tempDictionary.items():
	
	if len(value) > 1:
		data.drop(value, axis = 1, inplace = True)
		
		value = [subValue.split('|') for subValue in value]
		print(len(value), value)
		
		value = [item for subValue in value for item in subValue]
		data['|'.join(sorted(list(set(value))))] = [float(value) for value in key]
		
		print(data.values.shape)

# %%

columnsToDelete = []
for column in list(data):
	tempArray = [value for value in data[column].values if np.isfinite(value)]
	
	if len(tempArray) == 1:
		columnsToDelete.append(column)

print(len(columnsToDelete))
data.drop(columnsToDelete, axis = 1, inplace = True)

# %%

ewmParameter = 1

# %%

trajectoriesSet, trajectoriesSmoothOriginal, trajectoriesRaw = {}, {}, {}
for column in list(data):
	
	if column == 'index':
		continue
	
	tempDF = data[[column]]
	
	if len(list(set([value for value in tempDF[column].values if np.isfinite(value)]))) == 1:
		continue
	
	tempDF.index = pd.to_datetime(tempDF.index, yearfirst = True)
	
	tempDF = tempDF.resample('W').agg(np.nansum)
	tempDF.replace(0, np.nan, inplace = True)
	
	firstIndex = tempDF.first_valid_index()
	lastIndex = tempDF.last_valid_index()
	tempDFCut = tempDF.loc[firstIndex: lastIndex]
	
	tempDFCut = tempDFCut.ewm(span = np.max([1, len(tempDFCut.index) * ewmParameter])).mean()
	tempDF = tempDF.ewm(span = np.max([1, len(tempDFCut.index) * ewmParameter])).mean()
	
	trajectoriesSet[column] = [list(robust_scale(tempDFCut[column].values))]
	trajectoriesSmoothOriginal[column] = list(tempDF[column].values)
	trajectoriesRaw[column] = list(data[column].fillna(0))

# %%

maxLength = max([len(value[0]) for _, value in trajectoriesSet.items()])

# %%

trajectoriesSetProcessed = {}
for key, value in trajectoriesSet.items():
	value = value[0]
	
	if len(value) == maxLength:
		trajectoriesSetProcessed[key] = np.array(value).reshape(1, len(value))
		continue
	
	oldScale = np.arange(0, maxLength, maxLength / len(value))
	
	oldScale = oldScale[: min([len(oldScale), len(value)])]
	value = value[: min([len(oldScale), len(value)])]
	
	try:
		interpolationFunction = interpolate.interp1d(oldScale, value)
	
	except:
		print(value)
		continue
	
	cutOff = 0
	while True:
		newScale = np.linspace(0, maxLength - cutOff, maxLength)
		
		try:
			value = interpolationFunction(newScale)
			break
		
		except:
			cutOff += 1
	
	trajectoriesSetProcessed[key] = np.array(value).reshape(1, len(value))

# %%

trajectories = deepcopy(trajectoriesSetProcessed)
trajectoriesValues = np.array([value[0] for value in list(trajectories.values())])
trajectoriesKeys = list(trajectoriesSetProcessed.keys())

# %%

dm, _ = spearmanr(trajectoriesValues, axis = 1)
dm = -dm + 1

# %%

distanceMatrixDictionary = {}
for index1, filter1 in enumerate(trajectoriesKeys):
	for index2, filter2 in enumerate(trajectoriesKeys):
		
		if index1 >= index2:
			continue
		
		unionFilter = tuple([filter1, filter2])
		sorted(unionFilter)
		
		if unionFilter not in distanceMatrixDictionary.keys():
			distanceMatrixDictionary[unionFilter] = dm[index1][index2]

# %%

THRESHOLD = 0.75

iteration = 1
while True:
	indicesDictionary = {value: index for index, value in enumerate(trajectories.keys())}
	
	seen = []
	for index1, (filter1, trajectory1) in enumerate(trajectories.items()):
		tempArray = []
		
		for index2, (filter2, trajectory2) in enumerate(trajectories.items()):
			
			if index1 >= index2:
				continue
			
			else:
				
				if sorted([indicesDictionary.get(filter1), indicesDictionary.get(filter2)]) in seen:
					continue
				
				seen.append(sorted([indicesDictionary.get(filter1), indicesDictionary.get(filter2)]))
				
				unionFilter = tuple([filter1, filter2])
				
				if unionFilter in distanceMatrixDictionary.keys():
					continue
				
				metric = []
				
				for subItem1 in trajectory1:
					
					for subItem2 in trajectory2:
						
						try:
							metric.append(-spearmanr(subItem1, subItem2)[0] + 1)
						
						except:
							print(123)
				
				metric = max(metric)
				distanceMatrixDictionary[unionFilter] = metric
	
	minValue = np.nanmin(list(distanceMatrixDictionary.values()))
	print(minValue)
	
	if minValue > THRESHOLD:
		print(minValue, THRESHOLD)
		break
	
	minIndices = [key for key, value in distanceMatrixDictionary.items() if value == minValue]
	
	for minIndex in minIndices:
		
		if any(value not in trajectories.keys() for value in minIndex):
			trajectories = {key: value for key, value in trajectories.items() if key not in minIndex}
			distanceMatrixDictionary = {key: value for key, value in distanceMatrixDictionary.items()
			                            if all(value in trajectories.keys() for value in key)}
			
			continue
		
		trajectoryGroup = np.concatenate([trajectories.get(value) for value in minIndex])
		print(minIndex, trajectoryGroup.shape)
		
		trajectories = {key: value for key, value in trajectories.items() if key not in minIndex}
		
		trajectories[minIndex] = trajectoryGroup
		
		distanceMatrixDictionary = {key: value for key, value in distanceMatrixDictionary.items()
		                            if all(value in trajectories.keys() for value in key)}
	
	print(iteration, 'finished!')
	iteration += 1
	
	if len(list(trajectories.keys())) == 1:
		break

import sys

sys.exit(0)

# %%

minIndices = np.where(distanceMatrix == minValue)

# %%

distanceMatrix

# %%

from collections import Counter

print(Counter(list(minIndices)))

# %%

for key, value in trajectories.items():
	print(key, len(value))

# %%

clusterNames = sorted(list(set(trajectories.keys())))

nameColumn, idColumn = [], []
for clusterIndex, clusterName in enumerate(clusterNames):
	
	if len(clusterName) == 1:
		continue
	
	nameColumn.append('|'.join(clusterName))
	idColumn.append(clusterIndex + 1)

resultDF = pd.DataFrame()
resultDF['name'] = nameColumn
resultDF['id'] = idColumn

resultDF.to_csv('2.1.clusteringWeekly[correlation+interpolation].csv')

# %%

for clusterIndex, clusterName in enumerate(clusterNames):
	
	if len(clusterName) == 1:
		continue
	
	figure = make_subplots(rows = 3, cols = 1)
	colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(clusterName))]
	
	value = []
	for subKey in clusterName:
		value.append(trajectoriesSetProcessed.get((subKey,))[0])
	
	for index, subValue in enumerate(value):
		figure.add_trace(go.Scatter(x = list(range(0, len(subValue))), y = subValue,
		                            mode = 'lines', marker_color = colors[index], line = dict(width = 2.5), line_shape = 'spline'), row = 1, col = 1)
	
	#
	
	value = []
	for subKey in clusterName:
		value.append(trajectoriesSmoothOriginal.get(subKey))
	
	for index, subValue in enumerate(value):
		figure.add_trace(go.Scatter(x = list(range(0, len(subValue))), y = subValue,
		                            mode = 'lines', marker_color = colors[index], line = dict(width = 2.5), line_shape = 'spline'), row = 2, col = 1)
	
	value = []
	for subKey in clusterName:
		value.append(trajectoriesRaw.get(subKey))
	# value.append(trajectoriesSetProcessed.get((subKey, ))[0])
	
	for index, subValue in enumerate(value):
		figure.add_trace(go.Scatter(x = list(range(0, len(subValue))), y = subValue,
		                            mode = 'lines', marker_color = colors[index], line = dict(width = 2.5), line_shape = 'spline'), row = 3, col = 1)
	
	#
	
	figure.update_layout(title = 'Cluster ' + str(clusterIndex + 1), showlegend = False, height = 1200, width = 1200)
	figure.show()

# figure.write_image('images/cluster_' + str(clusterIndex + 1) + '.png')

# break

# %%
