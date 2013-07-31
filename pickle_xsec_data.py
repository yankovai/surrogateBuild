import xlrd
import pickle
import numpy as np

# get data for control rod OUT assembly
ws = xlrd.open_workbook('tmi7.xls').sheet_by_name('Sheet1')
tmi7out_columns = ws.row_values(0)

# Indices of entries transport1, transport2, absorption1, absorption2,
# nufission1, nufission2, downscatter, adf1, adf2 in tmi7out_columns
xsec_indices = [0,1,4,5,6,7,10,11,12]
tmi7out_columns = np.array(tmi7out_columns)[xsec_indices]

# data matrix is m-by-n
m = ws.nrows - 1
n = len(xsec_indices)

tmi7out = np.zeros([m,n])

for i in range(m):
    tmi7out[i,:] = np.array(ws.row_values(i+1))[xsec_indices]

tmi7out_mean = np.mean(tmi7out,0)
tmi7out_std = np.std(tmi7out,0)

# ---

# get data for control rod IN assembly
ws = xlrd.open_workbook('tmi7_cr.xls').sheet_by_name('Sheet1')
tmi7in_columns = ws.row_values(0)

# Indices of entries transport1, transport2, absorption1, absorption2,
# nufission1, nufission2, downscatter, adf1, adf2 in tmi7in_columns
xsec_indices = [0,1,4,5,6,7,10,11,12]
tmi7in_columns = np.array(tmi7in_columns)[xsec_indices]

# data matrix is m-by-n
m = ws.nrows - 1
n = len(xsec_indices)
tmi7in = np.zeros([m,n])

for i in range(m):
    tmi7in[i,:] = np.array(ws.row_values(i+1))[xsec_indices]

tmi7in_mean = np.mean(tmi7in,0)
tmi7in_std = np.std(tmi7in,0)

# ---

# get data for reflector
ws = xlrd.open_workbook('tmi_refl.xls').sheet_by_name('Sheet1')
tmi7rl_columns = ws.row_values(0)

# Indices of entries transport1, transport2, absorption1, absorption2,
# downscatter, adf1, adf2 in tmi7rl_columns
xsec_indices = [0,1,4,5,10,11,12]
tmi7rl_columns = np.array(tmi7rl_columns)[xsec_indices]

# data matrix is m-by-n
m = ws.nrows - 1
n = len(xsec_indices)
tmi7rl = np.zeros([m,n])

for i in range(m):
    tmi7rl[i,:] = np.array(ws.row_values(i+1))[xsec_indices]

tmi7rl_mean = np.mean(tmi7rl,0)
tmi7rl_std = np.std(tmi7rl,0)

# combine, process data and pickle
matrix_all_data = np.hstack((tmi7in,tmi7out,tmi7rl))
covmatrix = np.cov(matrix_all_data, rowvar=0)
meanvals = np.hstack((tmi7in_mean, tmi7out_mean, tmi7rl_mean))
stdvals = np.hstack((tmi7in_std, tmi7out_std, tmi7rl_std)) 

target_xsec_names =['{in_trsp1}',
                    '{in_trsp2}',
                    '{in_absp1}',
                    '{in_absp2}',
                    '{in_nufs1}',
                    '{in_nufs2}',
                    '{in_dwnsc}',
                    '{in_adfa1}',
                    '{in_adfa2}',
                    '{ot_trsp1}',
                    '{ot_trsp2}',
                    '{ot_absp1}',
                    '{ot_absp2}',
                    '{ot_nufs1}',
                    '{ot_nufs2}',
                    '{ot_dwnsc}',
                    '{ot_adfa1}',
                    '{ot_adfa2}',
                    '{rf_trsp1}',
                    '{rf_trsp2}',
                    '{rf_absp1}',
                    '{rf_absp2}',
                    '{rf_dwnsc}',
                    '{rf_adfa1}',
                    '{rf_adfa2}']


pickle.dump(covmatrix, open('covmatrix.dat','wb'))
pickle.dump(meanvals, open('mu.dat','wb'))
pickle.dump(stdvals, open('std.dat','wb'))
pickle.dump(target_xsec_names, open('target_xsec_names.dat','wb'))
    

