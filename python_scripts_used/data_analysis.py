import numpy as np
import scipy
import json
import matplotlib.pyplot as plt
from sys import platform
import os

#Creeëren van de benodigde numpy arrays met de data

if platform == "linux" or platform == "linux2" or platform == "darwin":
    HMC_directory = "../HMC_DATA"
    Heatbath_directory = "../HEATBATH_DATA"
    MH_directory = "../MH_DATA"
elif platform == "win32":
    HMC_directory = "..\HMC_DATA"
    Heatbath_directory = "..\HEATBATH_DATA"
    MH_directory = "..\MH_DATA"

#HMC
#INDEX 0: lambda
#INDEX 1: width (3-17)
#Vanaf dan zijn het allemaal lijsten met waardes; van die lijsten:
#INDEX 0: Kappa
#INDEX 1: Acceptance rate
#INDEX 2: Average field value
#INDEX 3: Error
#INDEX 4: Correlation time (sweeps)
#INDEX 5: Runtime (seconds)

HMC_DATA = np.empty((3,15,11,6))

for file in os.scandir(HMC_directory):
    if file.is_file():
        with open(file.path) as f:                
            data_dict = json.load(f)
            w = data_dict["parameters"]["w"]
            l = data_dict["parameters"]["l"]
            HMC_DATA[int(l*2 - 2)][w-3] = np.array(data_dict["full_results"])

#Heatbath
#INDEX 0: lambda
#INDEX 1: width (3-12)
#Vanaf dan zijn het allemaal lijsten met waardes; van die lijsten:
#INDEX 0: Kappa
#INDEX 1: Average field value
#INDEX 2: Error
#INDEX 3: Correlation time (sweeps)
#INDEX 4: Runtime (seconds)

HEATBATH_DATA = np.empty((3,10,11,5))

for file in os.scandir(Heatbath_directory):
    if file.is_file():
        with open(file.path) as f:                
            data_dict = json.load(f)
            w = data_dict["parameters"]["w"]
            l = data_dict["parameters"]["l"]
            HEATBATH_DATA[int(l*2 - 2)][w-3] = np.array(data_dict["full_results"])

#MH
#INDEX 0: lambda
#INDEX 1: width (3-10)
#Vanaf dan zijn het allemaal lijsten met waardes; van die lijsten:
#INDEX 0: Kappa
#INDEX 1: Acceptance rate
#INDEX 2: Average field value
#INDEX 3: Error
#INDEX 4: Correlation time (sweeps)
#INDEX 5: Runtime (seconds)


MH_DATA = np.empty((3,8,11,6))

for file in os.scandir(MH_directory):
    if file.is_file():
        with open(file.path) as f:                
            data_dict = json.load(f)
            w = data_dict["parameters"]["w"]
            l = data_dict["parameters"]["l"]
            MH_DATA[int(l*2 - 2)][w-3] = np.array(data_dict["full_results"])

#Check wat de resultaten voor w: 17 HMC zijn:

# plt.errorbar([k[0] for k in HMC_DATA[1][-1]], [k[2] for k in HMC_DATA[1][-1]], yerr= [k[3] for k in HMC_DATA[1][-1]], fmt = '-o')
# plt.xlabel(r"$\kappa$")
# plt.ylabel(r"$|m|$")
# plt.title(r"Absolute field average on $17^4$ lattice, $\lambda = 1.5$")
# plt.show()


#Print de runtime/cortime ratio voor de hoogste lattice size die die aankon:
def meas_time_check(w_check, lamb):
    ratio = []
    for r in HMC_DATA[lamb][w_check -3]:
        ratio.append(r[-1]/int((r[-2]+ 1)/2))  
    HMC = np.mean(ratio)
    if w_check < len(HEATBATH_DATA[lamb]) + 3:
        ratio = []
        for r in HEATBATH_DATA[lamb][w_check -3]:
            ratio.append(r[-1]/int((r[-2]+ 1)/2))
        Heatbath = np.mean(ratio) 
    else:
        Heatbath = np.inf    

    if w_check < len(MH_DATA[lamb]) + 3 :
        ratio = []
        for r in MH_DATA[lamb][w_check -3]:
            ratio.append(r[-1]/int((r[-2]+ 1)/2)) 
        MH = np.mean(ratio)
    else:
        MH = np.inf
    return [HMC, Heatbath, MH]
def plotting_per_lamb_w_show(lamb):
    values = np.array([meas_time_check(w,lamb) for w in range(3,18)])
    values_HMC = values[:,0]
    values_Heatbath = values[:,1][:10]
    values_MH = values[:,2][:8]

    lamb = (lamb + 2)/2

    plt.plot([index + 3 for index, _ in enumerate(values_HMC)], values_HMC, label = r"HMC with $\lambda$ = " + str(lamb), color = (0.2 + 0.2 * lamb, 0.3, 0.5))
    plt.plot([index + 3 for index, _ in enumerate(values_MH)], values_MH, label = r"MH with $\lambda$ = " + str(lamb), color = (0.5 + 0.2 * lamb,0.3,0.2))
    plt.plot([index + 3 for index, _ in enumerate(values_Heatbath)], values_Heatbath, label = r"Heatbath with $\lambda$ = " + str(lamb), color = (0.3 + 0.2 * lamb,0.5,0.2))

def plot_all_lamb():
    plotting_per_lamb_w_show(0)
    plotting_per_lamb_w_show(1)
    plotting_per_lamb_w_show(2)
    plt.legend()
    plt.title("estimated time per measurement")
    plt.xlabel("lattice size")
    plt.ylabel("mean(run_time/measurement_sweeps) (s)")
    plt.show()

plot_all_lamb()