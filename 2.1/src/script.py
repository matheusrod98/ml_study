# Import the libraries.
import pandas as pd
import matplotlib.pyplot as plt

# Loads the dataset.
data = pd.read_csv ('../data/iris.csv')

# Show the first part of the dataset.
# print (data.head ())

# Prints the number of lines and columns of the dataset.
# print (data.shape)

# Show the datatypes of the examples.
# print (data.dtypes)

# Create a dataframe with only the petals data.
petals_data = data [['petal_length','petal_width']]
# print (petals_data.head ())

# Remove the Iris prefix on the name of the species.
# data ['species'] = data ['species'].str.replace ("Iris-", "")
# print (data.head ())

# Another way to remove the 'Iris-' prefix.
data ['species'] = data ['species'].apply (lambda r: r.replace ("Iris-", ""))
# print (data.head ())

# Count the number of each species.
# print (data ['species'].value_counts ())

# Show info about the data.
# print (data.describe ())

# print (data.mean ())

# Show mean, std and median of each column.
# print (data.mean ())
# print (data.median ())
# print (data.std ())

# print (data.groupby ('species').mean ())

# Show a table with all the stats of the attributes.

# result = data.groupby ('species').agg(
        # {
            # x: ['median', 'mean', 'std'] for x in data.columns if x != 'species'
        # }
    # )
# print (result)

histgram = data ['petal_length'].plot.hist (bins = 20)
histgram.set (
        title = "Petal lenght distribution",
        xlabel = "Petal lenght (cm)",
        ylabel = "Number of examples"
        )
plt.show ()

scatter = data.plot.scatter ('petal_width', 'petal_length')
scatter.set (
        title = "Petal lenght dispersion",
        xlabel = "Petal width (cm)",
        ylabel = "Petal lenght (cm)"
        )
plt.show ()

attributes = data.iloc [:, :-1]
labels = data.iloc [:, -1]
classes = data ['species'].unique ().tolist ()
color_map = ['red', 'green', 'blue']
example_colors = [color_map [classes.index (r)] for r in labels]

pd.plotting.scatter_matrix (
        attributes,
        c = example_colors,
        figsize = (11, 11),
        marker = 'o',
        s = 30,
        alpha = 0.5,
        diagonal = 'hist',
        hist_kwds = {'bins': 20}
    )
plt.show ()

x_axis = 'sepal_length'
y_axis = 'petal_length'
z_axis = 'petal_width'
figure = plt.figure (figsize = (15, 12))
plot = figure.add_subplot (111, projection = '3d')
plot.scatter (
        data [x_axis],
        data [y_axis],
        data [z_axis],
        c = example_colors,
        marker = 'o',
        s = 40,
        alpha = 0.5,
    )
plt.show ()
