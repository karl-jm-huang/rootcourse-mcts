import os

f_path = "./test_data/Abnormalytime_forecast_PV_table"
r_path = "./test_data/Abnormalytime_real_PV_table"
files = os.listdir(f_path)

M = 100
PT = 0.75

for f in files:
    forecast = f_path+'/'+f
    real = r_path+'/'+f
    cmd = 'python version3.py '+str(M)+' '+str(PT)+' '+forecast+' '+real
    os.system(cmd)
    break