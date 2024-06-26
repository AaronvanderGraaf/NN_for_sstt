import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit_function(x, a):
    return a * x**2

def Plot_Fit(fit_result, wc, xs, string):
    plt.figure()
    plt.scatter(wc, xs, c='red', marker='o', label='uu')
    #plt.plot(wc, fit_function(wc, *fit_result), c='black', label=string + " Parameter = " + str(np.round(fit_result[0], 2)))
    wc_range = np.linspace(wc.min(), wc.max(), 100)
    plt.plot(wc_range, fit_function(wc_range, *fit_result), c='blue', label=string + " Parameter = " + str(np.round(fit_result[0], 2)))
    plt.xlabel('WC')
    plt.ylabel('cross section (pb)')
    plt.legend()
    plt.savefig(string+".png")
    #plt.show()

def Format_Results(limit_array, string):
    result = np.round(np.array([limit_array[0], abs(limit_array[0]-limit_array[1]), abs(limit_array[0]-limit_array[2])]), precision)

    print(string+ f' = {result[0]:.5f}_(-{result[1]:.5f})_(+{result[2]:.5f})')

    return result

# Settings
plotting=True
precision = 5
nom_signal_norm = 1.94 + 0.048
Final_Norm = 1# set to 1 if no signal norm otherwise 0.4*0.05 or the according value

tt_xs = 0.0973
tbartbar_xs = 0.002418

weight_mc_tt = 0.1006
weight_mc_tbartbar = 0.002528

#Inject new mu values here
nom_mu = 0.6225

# Old Limits
#uu_mu = np.array([4.206, 3.03, 6.2])
#qu1_mu = np.array([6.798, 4.898, 9.847])
#qu8_mu = np.array([7.392, 5.326, 10.7])

# Newer Limit with Signal Norm
#uu_mu = np.array([4.201, 2.847, 6.338])
#qu1_mu = np.array([6.824, 4.677, 10.18])
#qu8_mu = np.array([7.418, 5.085, 11.06])

# New Limits without Signal Norm (Full Asimov)
#uu_mu = np.array([0.2101, 0.1424, 0.3169])
#qu1_mu = np.array([0.3395, 0.2326, 0.5067])
#qu8_mu = np.array([0.3692, 0.2530, 0.5508])

# New Limits without Signal Norm (hybrid-Asimov)
#uu_mu = np.array([0.2323, 0.1585, 0.3483])
#qu1_mu = np.array([0.3663, 0.2517, 0.5445])
#qu8_mu = np.array([0.3969, 0.2728, 0.5900])

#New Limits without Signal Norm January 08.01.2024
uu_mu = np.array([0.1706, 0.1192, 0.2461])
qu1_mu = np.array([0.2314, 0.1627, 0.3341])
qu8_mu = np.array([0.2483, 0.1745, 0.3587])

# New Limit Stat Only
#uu_mu = np.array([0.1277, 0.0893, 0.1850])

# New Limit Feburary with new Root version with Norm SSTT Start value set to approx 0 
uu_mu = np.array([0.121866, 0.085313, 0.176841])

# New Limit Feburary with new Root version with Norm SSTT Start value set to approx 0 AND StatOnly!!!
#uu_mu = np.array([0.1115, 0.0776, 0.1625])

uu_xs_norm_tt = np.array([133.8, 301.1, 535.2, 836.3]) * weight_mc_tt
qu1_xs_norm_tt = np.array([99.11, 194.3, 396.4, 892.0]) * weight_mc_tt
qu8_xs_norm_tt = np.array([92.05, 368.2, 575.3, 828.5]) * weight_mc_tt

uu_xs_norm_tbartbar = np.array([146.1, 328.8, 584.6, 913.4]) * weight_mc_tbartbar
qu1_xs_norm_tbartbar = np.array([96.39, 188.9, 385.6, 867.5]) * weight_mc_tbartbar
qu8_xs_norm_tbartbar = np.array([96.85, 387.4, 605.3, 871.7]) * weight_mc_tbartbar

uu_xs_norm = uu_xs_norm_tt + uu_xs_norm_tbartbar
qu1_xs_norm = qu1_xs_norm_tt + qu1_xs_norm_tbartbar
qu8_xs_norm = qu8_xs_norm_tt + qu8_xs_norm_tbartbar

uu_xs_norm_final = uu_xs_norm * Final_Norm
qu1_xs_norm_final = qu1_xs_norm * Final_Norm
qu8_xs_norm_final = qu8_xs_norm * Final_Norm

wc_uu = np.array([0.02, 0.03, 0.04, 0.05])
wc_qu1 = np.array([0.05, 0.07, 0.1, 0.15])
wc_qu8 = np.array([0.1, 0.2, 0.25, 0.3])

nom_xs = nom_signal_norm * nom_mu

uu_wc_limit = []
qu1_wc_limit = []
qu8_wc_limit = []

for i in range(uu_mu.size):

    uu_xs = uu_xs_norm
    qu1_xs = qu1_xs_norm
    qu8_xs = qu8_xs_norm

    fit_uu, _ = curve_fit(fit_function, wc_uu, uu_xs, p0=1.0)
    fit_qu1, _ = curve_fit(fit_function, wc_qu1, qu1_xs, p0=1.0)
    fit_qu8, _ = curve_fit(fit_function, wc_qu8, qu8_xs, p0=1.0)

    if plotting == True:
        Plot_Fit(fit_uu, wc_uu, uu_xs, "Fit_uu")
        Plot_Fit(fit_qu1, wc_qu1, qu1_xs, "Fit_qu1")
        Plot_Fit(fit_qu8, wc_qu8, qu8_xs, "Fit_qu8")

    uu_wc_limit.append(np.round(np.sqrt(uu_xs_norm_final[0]*uu_mu[i]/fit_uu[0]), precision))
    qu1_wc_limit.append(np.round(np.sqrt(qu1_xs_norm_final[0]*qu1_mu[i]/fit_qu1[0]), precision))
    qu8_wc_limit.append(np.round(np.sqrt(qu8_xs_norm_final[0]*qu8_mu[i]/fit_qu8[0]), precision))

Format_Results(uu_wc_limit, "Cuu: ")
Format_Results(qu1_wc_limit, "Cqu1:")
Format_Results(qu8_wc_limit, "Cqu8:")