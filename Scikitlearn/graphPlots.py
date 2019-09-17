# plotting functions available in matplotlib package
import matplotlib
# if using a virtual environment such as conda  environment set backend rendering engine to TkAgg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# x-axis
# x = [0, 1, 2, 3, 4, 5, 6]
x = np.arange(-10, 10, 0.5)
# y-axis y = e^(x)
y = np.exp(x)
y2 = np.exp(-x)
# y = [0, 1, 2, 3, 4, 5, 6]


# customize the graph and plot the points or you can simply plot the graph using the plot(x, y) function
plt.plot(x, y, color='green', linestyle='dashed', linewidth=2, marker='o', markerfacecolor='blue')

# setting x and y axis range
plt.ylim(-1,10)
plt.xlim(-10,10)

# add labels and title
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('exponent plots')
# plt.show()

# use a new figure to draw subplots

# set style for plot
# you can use ggplot to stack subplots horizontally
plt.style.use('fivethirtyeight')

# create new figure
sub_plot_figure = plt.figure()

# define subplots and their positions
# first number defines number of rows (n) i.e 2 rows
# second number defines number of columns (m) i.e. 2 columns
# third number defines location of plot in nxm grid
plt1 = sub_plot_figure.add_subplot(221)
plt2 = sub_plot_figure.add_subplot(222)
plt3 = sub_plot_figure.add_subplot(223)
plt4 = sub_plot_figure.add_subplot(224)

plt1.plot(x, y, color='r')
plt1.set_title('exponent')
plt2.plot(x, y2, color='b')
plt2.set_title('negative exponent')

# adjust space between subplots
sub_plot_figure.subplots_adjust(hspace=0.5, wspace=0.5)

plt.show()