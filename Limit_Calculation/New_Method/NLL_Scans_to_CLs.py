import numpy as np
import matplotlib.pyplot as plt
import re, sys, yaml
import atlas_mpl_style as ampl
ampl.use_atlas_style()

atlas_label_status="Internal"

# Check if a filename is provided as a command-line argument
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    print("Please provide a filename as a command-line argument.")
    sys.exit(1)

# Read input from YAML file
with open(filename, 'r') as file:
    data = yaml.safe_load(file)

# Extract X and minusdeltaNLL values from the YAML data
X = np.array([entry['X'] for entry in data])
minusdeltaNLL = np.array([entry['minusdeltaNLL'] for entry in data])

# Calculate the minimum negative log-likelihood value
min_nll = np.min(minusdeltaNLL)

# Calculate the y-axis values corresponding to the 68% and 95% CLs
y_68 = round(min_nll + 0.5, 4)  # Add 0.5 to the minimum value and round to 4 decimal places
y_95 = round(min_nll + 1.92, 4)  # Add 1.92 (corresponds to 2*sqrt(2)) to the minimum value and round to 4 decimal places

# Find the parameter values (x-axis values) corresponding to the y-axis values using linear interpolation
x_68_pos = np.interp(y_68, minusdeltaNLL[20:40], X[20:40])
x_95_pos = np.interp(y_95, minusdeltaNLL[20:40], X[20:40])

x_68_neg = np.interp(y_68, minusdeltaNLL[0:20][::-1], X[0:20][::-1])
x_95_neg = np.interp(y_95, minusdeltaNLL[0:20][::-1], X[0:20][::-1])

x_upper_lim_pos = abs(np.round(np.interp(4, minusdeltaNLL[20:40], X[20:40]), 3))
x_upper_lim_neg = x_upper_lim_pos * -1

# Extract the x-label from the YAML file name
match = re.search(r'NLLscan_(\w+)\.yaml', filename)
if match:
    x_label = match.group(1)
else:
    x_label = 'X'  # Default label if no match is found

if x_label == "cuu":
    x_variable_latex = r'$c_{uu}$'
if x_label == "cqu1":
    x_variable_latex = r'$c_{qu}^{1}$'
if x_label == "cqu8":
    x_variable_latex = r'$c_{qu}^{8}$'

# Plotting the data points and interpolation lines
plt.plot(X, minusdeltaNLL, color='black', label='Data', marker='o', linestyle='-', linewidth=1)
plt.axhline(y=y_68, color='blue', linestyle='--', label='68% CL')
plt.axhline(y=y_95, color='green', linestyle='--', label='95% CL')

# Add text annotations for the confidence levels
plt.text(x_upper_lim_pos*0.99, y_68, r'$1\sigma$', color='blue', ha='right', va='bottom', transform=plt.gca().transData)
plt.text(x_upper_lim_pos*0.99, y_95, r'$2\sigma$', color='green', ha='right', va='bottom', transform=plt.gca().transData)


plt.gca().ticklabel_format(axis='x', style='plain', useOffset=False)
# Set the y-range to show the confidence levels clearly
plt.ylim(bottom=-0.1, top=8)  # Adjust the upper limit as needed
# Set the x-range to include the parameter values for the confidence levels
plt.xlim(left=X.min(), right=X.max())  # Adjust the limits as needed

ampl.set_xlabel(x_variable_latex)  
ampl.set_ylabel(r'$\Delta$ NLL')
legend_text = [
    'NLL Scan values',
    '68% CL '+x_variable_latex+' = [{:.4f}, {:.4f}]'.format(x_68_neg, x_68_pos),
    '95% CL '+x_variable_latex+' = [{:.4f}, {:.4f}]'.format(x_95_neg, x_95_pos)
]
ampl.plot.draw_atlas_label(0.04, 0.96, status=atlas_label_status)
plt.legend(legend_text, bbox_to_anchor=(0.05, 0.92), loc='upper left')
#plt.show()
plt.savefig("plots/CLs_"+x_label+".png")

print("95% CLs Limit on " + x_label + " is [{:.4f}, {:.4f}]".format(x_95_neg, x_95_pos))


