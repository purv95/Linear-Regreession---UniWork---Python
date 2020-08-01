import numpy as np
from matplotlib import pyplot as plt

################# Reading data from csv file #################
cleanData = open('cleanData.csv', 'r')
cleanData.readline()
dataStr = cleanData.readlines()
newData = []
for line in dataStr:
    ind = line.find(',')
    interest = float(line[:ind])
    line = line[ind+1:]
    
    ind = line.find(',')
    bias = float(line[:ind])
    line = line[ind+1:]

    ind = line.find(',')
    value = float(line[:ind])
    line = line[ind+1:]

    newData.append([interest, bias, value])

cleanData.close()

################# Spliting data into X and Y #################
newData = np.array([newData])
newData = newData[0,:,:]
X = newData[:, :2] #Selecting the first 2 columns and all samples
Y = newData[:, 2:] #Selecting the last column and all samples

sumXY = 0
sumX = 0
numSamples = len(newData)

for i in range(len(X)):
    sumXY += X[i,:]*Y[i] #Summing the numberator terms
    sumX += X[i,0]**2 #Summing all the denominator terms

params1 = sumXY/sumX #Getting the optimal parameters
Yn = np.dot(X, params1) #Multiplying matrix
error = Yn.T - Y.T #Calculating error
MSE = (1/numSamples)*np.sum(error**2)
RMSE = np.sqrt(MSE)
print("\nSLR:")
print("RMSE: ", RMSE)
print("The optimal parameters are: \n", params1)


plt.figure()
plt.title("SLR")
plt.plot(X[:,0], Y, "X", label='Real Datapoints')
plt.plot(X[:,0], Yn, label='Predicted Linear Model')
plt.ylabel('Bitcoin Value')
plt.xlabel('Bitcoin Interest')
plt.legend()


################# Selecting 5 samples for classes part #################
print("\nWith 5 Data points for classes: ")
X = newData[114:119, :2] #Selecting the first 2 columns and 5 samples
Y = newData[114:119, 2:] #Selecting the last column and 5 samples

print('X: ',X)
print('Y: ',Y)

sumXY = 0
sumX = 0

for i in range(len(X)):
    sumXY += X[i,:]*Y[i]
    sumX += X[i,0]**2

# print("\n5 samples: \n")
# print("\n5 X vector values: \n", X)
# print("\n5 Y vector values: \n", Y)

params1 = sumXY/sumX
Yn = np.dot(X, params1)
error = Yn.T - Y.T
MSE = (1/numSamples)*np.sum(error**2)
RMSE = np.sqrt(MSE)

print("RMSE: ", RMSE)
print("The optimal parameters are: \n", params1)
plt.figure()
plt.title("5 Data points")
plt.plot(X[:,0], Y, "X", label='Real Datapoints')
plt.plot(X[:,0], Yn, label='Predicted Linear Model')
plt.ylabel('Bitcoin Value')
plt.xlabel('Bitcoin Interest')
plt.legend()
plt.show()
