# Pandas is built on top of numpy.
# Provides efficient implementation of dataframe
# Dataframes are multidimensional arrays with rows and column labels.


# Series
data = pd.Series([0.25, 0.5, 0.75, 1.0])
data
0 0.25
1 0.5
2 0.75
3 1.0

data.values

data.index

# indexing
data[1]

data[1:3]

# Series as a specialized dictionary
population_dict = {'California': 38332521,
        'Texas': 26448193,
        'New York': 19651127,
        'Florida': 19552860,
        'Illinois': 12882135}
population = pd.Series(population_dict)
population


area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
                     'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area

# Dataframe object
# Dataframe is multidimensional array
states = pd.DataFrame({'population': population, 'area': area})
states.index
# to print all indicies of dataframe


states['area'] will give the column


# constructing dataframe
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])


# Data indexing and slicing

# indexing on series
import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                         index=['a', 'b', 'c', 'd'])
data
data['a':'c']
data[(data>0.3) & (data<0.8)]


# Use of loc , iloc and ix
data = pd.Series(['a','b','c'], index=[1,3,5])
data[1]   # use explicit index

data.loc[1]  # always use explicit index

data.iloc[1]  # prints the second element, implicit python style index

data.iloc[:3,:2]
data.loc[:'Illinois',:'pop']

data.ix[:3,:'pop']

# we can combine fancy indexing
data.loc[data.density > 100, ['pop', 'density'])

