import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('student_cgpa.csv')
print("The shape of data: ",df.shape)
print(df.head())

plt.scatter(
    df['CGPA'], # as x-axis
    df['IQ'] # as y-axis
)
plt.show()

# now through elbow method we find how many cluster we will make
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit_predict(df)
    wcss.append(km.inertia_)

print(wcss)
plt.plot(range(1,11),wcss)  # x-axis-->range and y-axis-->wcss
plt.show()  # from graph we see that out steepness rate become slow at 4 we make 4 clusters

x = df.iloc[:,:].values # take all rows and columns
km = KMeans(n_clusters=3)
y_mean = km.fit_predict(x)

print("Y_Mean: ",y_mean)

x[y_mean == 0,0] # y_mean == 0 it will select all rows that comes in cluster 0 and after comma 0 represent selct 1st value of cluster 0 
# if it ==1 than select all that rows which include cluster1
print(x)

# Now plot using above logic doing for 4 clusters

plt.scatter(
    x[y_mean == 0,0],
    x[y_mean == 0,1],
    color = 'blue'
)

plt.scatter(
    x[y_mean == 1,0],
    x[y_mean == 1,1],
    color = 'red'
)

plt.scatter(
    x[y_mean == 2,0],
    x[y_mean == 2,1],
    color = 'green'
)


plt.show()
# blue cluster students normal cgpa and IQ level
# red has low CGPA and Low IQ level
# Green High CGPA and High IQ level

# So from above data teacher need to concentrate on red and blue cluster student
# because they need more to study
