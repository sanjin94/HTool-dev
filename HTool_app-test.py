import sys
sys.path.append('icons')
sys.path.append('data')
sys.path.append('calc')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

import htool as ht
import input_processing as inp
import WScrape as WS

test_materijal = ht.Material()
test_materijal.new_material('beton', cond=2.6, rho=2400, c=1000, l=0.135)
test_materijal.new_material('izolacija', cond=0.04, rho=40, c=1030, l=0.16)
test_otpor = ht.Resistance(test_materijal.layers, delta_x=0.005, delta_t=60)
R_mat, tau, R_bound, mesh = test_otpor.resistance_tau()

name = 'initial_notebook/TRSYS01_Public-beton.dat'
index = inp.RawData_index(name, '2021-12-15 06:00', '2021-12-22 06:00')
first, last = index.dfinit()
columns = index.cols()

series = inp.RawData_series(name, first, last, columns)
vectors = series.ex_vect()

indoor = vectors[0]
outdoor = vectors[1]
hf = vectors[4]

# Set initial temperature as stationary or as one temperature in all the layers
outdoor_init = np.array([15 for i in range(60*60*3)])
indoor_init = np.array([21 for i in range(60*60*3)])
initial = np.array([18 for i in range(len(tau))])
res_init = test_otpor.solve_he(R_mat, tau, R_bound, initial, indoor_init, outdoor_init)

initial = res_init[-1][1:len(res_init[-1])-1]
results = test_otpor.solve_he(R_mat, tau, R_bound, initial, indoor, outdoor)
q_calc, Q_calc = test_otpor.q_Q(results, mesh)
#plt.plot(q_calc)
#plt.plot(hf)
#plt.show()

mse = metrics.mean_squared_error(hf, q_calc)
mae = metrics.mean_absolute_error(hf, q_calc)
r2 = metrics.r2_score(hf, q_calc)
print(f'mse: {mse}, mae: {mae}, Rsquared: {r2}')

test_materijal = ht.Material()
test_materijal.new_material('beton', cond=2.6, rho=2400, c=1000, l=0.135)
test_materijal.new_material('izolacija', cond=0.035, rho=40, c=1030, l=0.16)
U_cls = ht.U_heat_flux(test_materijal.layers)
U_val = U_cls.uval()
q_U, Q_U = U_cls.q_Q(U_val, indoor, outdoor)

plt.plot(q_calc)
plt.plot(hf)
#plt.plot(q_U)
plt.show()
Q_test = np.sum(hf)
print(Q_test, Q_U, Q_calc)

test_materijal = ht.Material()
#test_materijal.new_material('žbuka', cond=1, rho=1800, c=1000, l=0.02)
test_materijal.new_material('izolacija', cond=0.04, rho=40, c=1030, l=0.16)
test_materijal.new_material('beton', cond=2.6, rho=2400, c=1000, l=0.135)
#test_materijal.new_material('žbuka', cond=1, rho=1800, c=1000, l=0.02)
test_otpor = ht.Resistance(test_materijal.layers, delta_x=0.005, delta_t=60)
R_mat, tau, R_bound, mesh = test_otpor.resistance_tau()

initial = res_init[-1][1:len(res_init[-1])-1]
results_obrnuto = test_otpor.solve_he(R_mat, tau, R_bound, initial, indoor, outdoor)
q_calc_obrnuto, Q_calc_obrnuto = test_otpor.q_Q(results_obrnuto, mesh)

## Comparison insulation inside/outside
#plt.plot(q_calc)
#plt.plot(q_calc_obrnuto)
#plt.show()

print(Q_test, Q_U, Q_calc, Q_calc_obrnuto)

scrape_2021 = WS.WUscrape('Osijek', 2021)
city_str = scrape_2021.city_find()
temperature = scrape_2021.scrape(city_str)