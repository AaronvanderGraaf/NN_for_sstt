import ROOT, sys
import numpy as np
import matplotlib.pyplot as plt

# Check if a filename is provided as a command-line argument
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print("Please provide a filename as a command-line argument.")
    sys.exit(1)

# Find the indices of the underscores
underscore_indices = [i for i, char in enumerate(filename) if char == '_']

# Extract the substrings between the underscores
x_variable = filename[underscore_indices[0] + 1:underscore_indices[1]]
y_variable = filename[underscore_indices[1] + 1:underscore_indices[2]]

# Open the ROOT file
file = ROOT.TFile(filename) #NLLscan_cqu1_cqu8_curve.root

# Access the TGraph2D object
graph = file.Get("LHscan_2D_"+x_variable+"_"+y_variable)

# Get the number of entries in the TGraph2D
n_entries = graph.GetN()

# Retrieve the data using the TGraph2D array method
x_values = graph.GetX()
y_values = graph.GetY()
z_values = graph.GetZ()

# Convert the arrays to NumPy arrays
x_array = np.array(x_values)
y_array = np.array(y_values)
z_array = np.array(z_values)

# Save the arrays to a CSV file
output_filename = "data/data_"+x_variable+"_"+y_variable+".csv"
data = np.column_stack((x_array, y_array, z_array))
np.savetxt(output_filename, data, delimiter=',')

print("Arrays saved to", output_filename)
