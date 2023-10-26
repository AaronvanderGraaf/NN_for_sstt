import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import atlas_mpl_style as ampl
import warnings

# Suppress FutureWarning for non-tuple sequence indexing
warnings.filterwarnings("ignore", category=FutureWarning)

ampl.use_atlas_style()

atlas_label_status="Internal"

def Generate_Contour_Line(x_array, y_array, z_array, likelihood_threshold, x_distance, y_distance, neg_y=False):
    # Identify the region below the likelihood threshold as the 95% CL region
    if neg_y == True:
        y_array = abs(y_array)

    indices_95cl = np.where(z_array <= likelihood_threshold)

    # Get the x and y values corresponding to the 95% CL region
    x_95cl = x_array[indices_95cl]
    y_95cl = y_array[indices_95cl]

    # Create a list to store the contour points
    contour_points = []

    # Iterate over the x values
    for x_val in np.unique(x_95cl):
        # Find the maximum y value that fulfills the likelihood requirement for the given x value
        max_y_val = np.max(y_95cl[x_95cl == x_val])
        # Append the contour point (x, y) to the list
        contour_points.append((x_val, max_y_val))

    y_numbers = [t[1] for t in contour_points]

    for y_val in np.unique(y_95cl):
        if y_val not in y_numbers:
            max_x_val = np.max(x_95cl[y_95cl == y_val])
            contour_points.append((max_x_val, y_val))

    # Sort the contour points based on the x values and y values
    contour_points = sorted(contour_points, key=lambda x: (x[0], -x[1]))

    # Convert the contour points to NumPy arrays
    x_contour = np.array([point[0] for point in contour_points])
    y_contour = np.array([point[1] for point in contour_points])

    x_contour = np.insert(x_contour, 0, x_contour[0]-x_distance)
    y_contour = np.insert(y_contour, 0, y_contour[0])
    #x_contour = np.append(x_contour, x_contour[-1])
    #y_contour = np.append(y_contour, y_contour[-1]-y_distance)

    modified_x_contour = []
    modified_y_contour = []

    for i in range(len(x_contour)):
        if i!=0 and x_contour[i] - x_contour[i-1] != 0 and y_contour[i] - y_contour[i-1] != 0:
            modified_x_contour.append(x_contour[i-1])
            modified_y_contour.append(y_contour[i])
            modified_x_contour.append(x_contour[i])
            modified_y_contour.append(y_contour[i])
        else:
            modified_x_contour.append(x_contour[i])
            modified_y_contour.append(y_contour[i])
    if neg_y == True:
        modified_y_contour = (np.array(modified_y_contour) * -1)[::-1].tolist()
        modified_x_contour = np.array(modified_x_contour)[::-1].tolist()

    return modified_x_contour, modified_y_contour

# Check if a filename is provided as a command-line argument
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print("Please provide a filename as a command-line argument.")
    sys.exit(1)

# Find the indices of the underscores
underscore_indices = [i for i, char in enumerate(filename) if char == '_']

# Extract the substrings between the underscores
x_variable = filename[underscore_indices[-2] + 1:underscore_indices[-1]]
y_variable = filename[underscore_indices[-1] + 1:-4]  # Exclude the ".csv" extension

if x_variable == "cuu":
    x_variable_latex = r'$c_{uu}$'
if x_variable == "cqu1":
    x_variable_latex = r'$c_{qu}^{1}$'
if x_variable == "cqu8":
    x_variable_latex = r'$c_{qu}^{8}$'
if y_variable == "cuu":
    y_variable_latex = r'$c_{uu}$'
if y_variable == "cqu1":
    y_variable_latex = r'$c_{qu}^{1}$'
if y_variable == "cqu8":
    y_variable_latex = r'$c_{qu}^{8}$'

# Read the CSV file
data = np.loadtxt(filename, delimiter=',')

# Extract the columns from the data
x_array = data[:, 0]
y_array = data[:, 1]
z_array = data[:, 2]

# Find the minimum likelihood value in the scan
min_likelihood = np.min(z_array)

# Calculate the likelihood ratio (LR) for each point
likelihood_ratio = 2 * (z_array - min_likelihood)

# Convert LR to chi-square values with 2 degrees of freedom
chi_square = likelihood_ratio

# Find the likelihood threshold for 95% CL (chi-square threshold of 3.84)
likelihood_threshold = min_likelihood + 3
likelihood_threshold_68 = min_likelihood + 1.3

X = x_array.reshape(40,40)
Y = y_array.reshape(40,40)
Z = z_array.reshape(40,40)

fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figsize as needed

# Create a scatter plot with color bar
pcm = ax.pcolormesh(X, Y, Z, cmap='viridis')
plt.colorbar(pcm, label=r'$\Delta$ NLL')
ampl.set_xlabel(x_variable_latex)
ampl.set_ylabel(y_variable_latex)
plt.title('2D NLL Scan '+ x_variable_latex + ' vs ' + y_variable_latex)
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
plt.xticks(y=-0.01)
plt.yticks(x=-0.01)
# Create a custom formatter to set maximum decimals
formatter = ticker.FormatStrFormatter('%.3f')  # Set the number of decimals to 2
# Set the x-axis tick label format
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)
ampl.plot.draw_atlas_label(0.04, 0.96, status=atlas_label_status)

x_distance = X[1][0]-X[0][0]
y_distance = Y[0][1]-Y[0][0]


pos_mask = [y_array > 0]
neg_mask = [y_array < 0]

x_contour_pos, y_contour_pos = Generate_Contour_Line(x_array[pos_mask], y_array[pos_mask], z_array[pos_mask], likelihood_threshold, x_distance, y_distance)
x_contour_68_pos, y_contour_68_pos = Generate_Contour_Line(x_array[pos_mask], y_array[pos_mask], z_array[pos_mask], likelihood_threshold_68, x_distance, y_distance)
x_contour_neg, y_contour_neg = Generate_Contour_Line(x_array[neg_mask], y_array[neg_mask], z_array[neg_mask], likelihood_threshold, x_distance, y_distance, True)
x_contour_68_neg, y_contour_68_neg = Generate_Contour_Line(x_array[neg_mask], y_array[neg_mask], z_array[neg_mask], likelihood_threshold_68, x_distance, y_distance, True)

x_contour_neg.append(x_contour_pos[0])
y_contour_neg.append(y_contour_pos[0])
x_contour_68_neg.append(x_contour_68_pos[0])
y_contour_68_neg.append(y_contour_68_pos[0])

x_contour = np.concatenate([x_contour_pos, x_contour_neg])
y_contour = np.concatenate([y_contour_pos, y_contour_neg])
x_contour_68 = np.concatenate([x_contour_68_pos, x_contour_68_neg])
y_contour_68 = np.concatenate([y_contour_68_pos, y_contour_68_neg])

# Plot the contour line using the unique x and y values
plt.plot(x_contour+x_distance/2, y_contour+y_distance/2, color='red', linewidth=2, label='95% CL')
plt.plot(x_contour_68+x_distance/2, y_contour_68+y_distance/2, color='orange', linewidth=2, label='68% CL')

# Add legend
plt.legend()

# Show the plot
#plt.show()
plt.savefig("plots/2D_Limits_"+x_variable+"_"+y_variable+".png")