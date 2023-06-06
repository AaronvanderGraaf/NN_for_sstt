import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit_function(x, a):
    return a * x**2

def Plot_Fit(fit_result, wc, xs, string):
    plt.figure()
    plt.scatter(wc, xs, c='red', marker='o', label='uu')
    plt.plot(wc, fit_function(wc, *fit_result), c='black', label=string + " Parameter = " + str(np.round(fit_result[0], 2)))
    plt.xlabel('WC')
    plt.ylabel('cross section (pb)')
    plt.title('cuu')
    plt.legend()
    plt.show()

def Format_Results(limit_array, string):
    result = np.round(np.array([limit_array[0], limit_array[0]-limit_array[1], limit_array[0]-limit_array[2]]), precision)

    #print(string+"="+str(result[0])+"_{"+str(result[1])+"}^{+"+str(result[2])+"}")
    #print(f'={result[0]:.3f}_{result[1]:.3f}^{+{result[1]:.3f}}')
    print(string+ f' = {result[0]:.3f}_({result[1]:.3f})_(+{result[2]:.3f})')

    return result

# Settings
plotting=False
precision = 3
nom_signal_norm = 1.94 + 0.048
Final_Norm = 0.4 * 0.05

tt_xs = 0.0973
tbartbar_xs = 0.002418

weight_mc_tt = 0.1006
weight_mc_tbartbar = 0.002528

#Inject new mu values here
nom_mu = 0.6225
uu_mu = np.array([4.206, 3.03, 6.2])
qu1_mu = np.array([6.798, 4.898, 9.847])
qu8_mu = np.array([7.392, 5.326, 10.7])

uu_xs_norm_tt = np.array([133.8, 301.1, 535.2, 836.3]) * weight_mc_tt
qu1_xs_norm_tt = np.array([99.11, 194.3, 396.4, 892.0]) * weight_mc_tt
qu8_xs_norm_tt = np.array([92.05, 368.2, 575.3, 828.5]) * weight_mc_tt

uu_xs_norm_tbartbar = np.array([146.1, 328.8, 584.6, 913.4]) * weight_mc_tbartbar
qu1_xs_norm_tbartbar = np.array([96.39, 188.9, 385.6, 867.5]) * weight_mc_tbartbar
qu8_xs_norm_tbartbar = np.array([96.85, 387.4, 605.3, 871.7]) * weight_mc_tbartbar

uu_xs_norm = uu_xs_norm_tt + uu_xs_norm_tbartbar
qu1_xs_norm = qu1_xs_norm_tt + qu1_xs_norm_tbartbar
qu8_xs_norm = qu8_xs_norm_tt + qu8_xs_norm_tbartbar

print(uu_xs_norm)
print(qu1_xs_norm)
print(qu8_xs_norm)

uu_xs_norm_final = uu_xs_norm * Final_Norm
qu1_xs_norm_final = qu1_xs_norm * Final_Norm
qu8_xs_norm_final = qu8_xs_norm * Final_Norm

print(uu_xs_norm_final)
print(qu1_xs_norm_final)
print(qu8_xs_norm_final)

# Noemi Values
#uu_xs_norm = np.array([3.17, 7.13, 12.68, 19.81])
#qu1_xs_norm = np.array([1.46, 2.86, 5.88, 13.21])
#qu8_xs_norm = np.array([1.26, 5.02, 7.86, 11.31])

wc_uu = np.array([0.02, 0.03, 0.04, 0.05])
wc_qu1 = np.array([0.05, 0.07, 0.1, 0.15])
wc_qu8 = np.array([0.1, 0.2, 0.25, 0.3])


nom_xs = nom_signal_norm * nom_mu
uu_wc_limit = []
qu1_wc_limit = []
qu8_wc_limit = []

for i in range(uu_mu.size):

    uu_xs = uu_xs_norm * uu_mu[i]
    qu1_xs = qu1_xs_norm * qu1_mu[i]
    qu8_xs = qu8_xs_norm * qu8_mu[i]

    fit_uu, _ = curve_fit(fit_function, wc_uu, uu_xs, p0=1.0)
    fit_qu1, _ = curve_fit(fit_function, wc_qu1, qu1_xs, p0=1.0)
    fit_qu8, _ = curve_fit(fit_function, wc_qu8, qu8_xs, p0=1.0)

    if plotting == True:
        Plot_Fit(fit_uu, wc_uu, uu_xs, "Fit uu")
        Plot_Fit(fit_qu1, wc_qu1, qu1_xs, "Fit qu1")
        Plot_Fit(fit_qu8, wc_qu8, qu8_xs, "Fit qu8")

    uu_wc_limit.append(np.round(np.sqrt(nom_xs/fit_uu[0]), precision))
    qu1_wc_limit.append(np.round(np.sqrt(nom_xs/fit_qu1[0]), precision))
    qu8_wc_limit.append(np.round(np.sqrt(nom_xs/fit_qu8[0]), precision))

Format_Results(uu_wc_limit, "Cuu: ")
Format_Results(qu1_wc_limit, "Cqu1:")
Format_Results(qu8_wc_limit, "Cqu8:")