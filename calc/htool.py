import numpy as np
import math
from scipy.sparse import diags
from scipy import linalg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import base64


# Material class defines a dictionary conditioning an information
# on material characteristics of each element layer
class Material:
  def __init__(self):
    self.layers = []

  def new_material(self, material_name, cond, rho, c, l):
    my_dict = {'material' : material_name,
              'conductivity' : cond,
              'density' : rho,
              'capacity' : c,
              'layer_length' : l}
    self.layers.append(my_dict)

# Defining Resistance matrix and input vector needed for equation
# solution of system R * T = t, where R is resistance matrix, T is
# solution vector of temperatures (T = t / R) and t is input vector
class Resistance:
  def __init__(self, layers, delta_x, delta_t, Rsi=0.13, Rse=0.04):
    self.layers = layers
    self.R = []
    self.tau = []
    self.dx = delta_x
    self.delta_t = delta_t
    self.Rsi = Rsi
    self.Rse = Rse
    self.mat_list = []
    self.l = 0
    self.R_i = 0
    self.R_e = 0

  def resistance_tau(self):
    mesh = []
    mesh_cumulative = []
    help = 0
    eps = 1e-5
    diag_c = []
    diag_w = [0]
    diag_e = []
    for i in range(len(self.layers)):
      min = help
      max = help + self.layers[i]['layer_length']
      conductivity = self.layers[i]['conductivity']
      density = self.layers[i]['density']
      capacity = self.layers[i]['capacity']
      self.mat_list.append([min, max, conductivity, density, capacity])
      help = max

    self.l = max
    mat_list = np.copy(self.mat_list)
    
    help = 0
    for i in range(len(mat_list)):
      if self.dx > (mat_list[i][1] - mat_list[i][0]):
        delta_x = (mat_list[i][1] - mat_list[i][0])
      else:
        delta_x = self.dx
      
      min = mat_list[i][0]
      max = mat_list[i][1]
      material = mat_list[i][2]

      firstl = True
      lastl = True

      for j in range(math.ceil((max - min) / delta_x)):
        help += delta_x

        if (i == (len(mat_list) - 1)) and (abs(help - mat_list[i][1]) < eps) and lastl:
          dx = dx_h
          Rw = dx / 2 / mat_list[-1][2] + delta_x / 2 / mat_list[-1][2]
          Re = dx / 2 / mat_list[-1][2] + self.Rse
          self.R_e = Re
          Ci = mat_list[-1][3] * mat_list[-1][4] * dx / self.delta_t + 1 / Rw + 1 / Re
          diag_w.append(-1 / Rw)
          diag_c.append(Ci)
          self.tau.append(mat_list[-1][3] * mat_list[-1][4] * dx / self.delta_t)
          ## loop check
          #print('inside last 1')
          #print('help: ', help)
          #print('dx =', dx, 'delta_x = ', delta_x)
          ## mesh check
          #print(dx)
          mesh.append(dx)
          break
        elif (i == (len(mat_list) - 1)) and (help - mat_list[i][1] > 0) and lastl:
          dx = delta_x - (help - mat_list[i][1])
          Rw = dx / 2 / mat_list[-1][2] + delta_x / 2 / mat_list[-1][2]
          Re = dx / 2 / mat_list[-1][2] + self.Rse
          self.R_e = Re
          Ci = mat_list[-1][3] * mat_list[-1][4] * dx / self.delta_t + 1 / Rw + 1 / Re
          diag_w.append(-1 / Rw)
          diag_c.append(Ci)
          self.tau.append(mat_list[-1][3] * mat_list[-1][4] * dx / self.delta_t)
          ## loop check
          #print('inside last 2')
          #print('help: ', help)
          #print('dx =', dx, 'delta_x = ', delta_x)
          ## mesh check
          #print(dx)
          mesh.append(dx)
          break


        if help == delta_x:
          Rw = help / 2 / mat_list[0][2] + self.Rsi
          self.R_i = Rw
          Re = help / mat_list[0][2]
          Ci = mat_list[0][3] * mat_list[0][4] * help / self.delta_t + 1 / Rw + 1 / Re
          diag_c.append(Ci)
          diag_e.append(-1 / Re)
          self.tau.append(mat_list[0][3] * mat_list[0][4] * help / self.delta_t)
          ## loop check
          #print('!!inside first')
          #print('help: ', help)
          ## mesh check
          #print(delta_x)
          mesh.append(delta_x)
          dx_h = delta_x
          continue

        if (help >= mat_list[i][0]) and j == 0 and i > 0: 
          dx = dx_h
          Rw = dx / 2 / mat_list[i-1][2] + delta_x / 2 / mat_list[i][2]
          Re = delta_x / mat_list[i][2]
          Ci = mat_list[i][3] * mat_list[i][4] * delta_x / self.delta_t + 1 / Rw + 1 / Re
          diag_w.append(-1 / Rw)
          diag_c.append(Ci)
          diag_e.append(-1 / Re)
          self.tau.append(mat_list[i][3] * mat_list[i][4] * delta_x / self.delta_t)
          help = mat_list[i][0] + delta_x
          ## loop check
          #print('!!inside b1')
          #print('help: ', help)
          #print('dx =', dx, 'delta_x = ', delta_x)
          ## mesh check
          #print(delta_x)
          mesh.append(delta_x)
          continue
        
        if (i < (len(mat_list)-1)) and (abs(help - mat_list[i][1]) < eps) and lastl:
          dx = mat_list[i][1] - (help - delta_x)
          dx_h = dx
          Rw = dx / 2 / mat_list[i][2] + delta_x / 2 / mat_list[i][2]
          Re = dx / 2 / mat_list[i][2] + delta_x / 2 / mat_list[i+1][2]
          Ci = mat_list[i][3] * mat_list[i][4] * dx / self.delta_t + 1 / Re + 1 / Rw
          diag_w.append(-1 / Rw)
          diag_c.append(Ci)
          diag_e.append(-1 / Re)
          self.tau.append(mat_list[i][3] * mat_list[i][4] * dx / self.delta_t)
          lastl = False
          ## loop check
          #print('!!inside b21')
          #print('help: ', help)
          #print('dx =', dx)
          #print(help, mat_list[i][1])
          ## mesh check
          #print(dx)
          mesh.append(dx)
          continue
        elif (i < (len(mat_list)-1)) and (help - mat_list[i][1] > 0) and lastl:
          dx = mat_list[i][1] - (help - delta_x)
          dx_h = dx
          Rw = dx / 2 / mat_list[i][2] + delta_x / 2 / mat_list[i][2]
          Re = dx / 2 / mat_list[i][2] + delta_x / 2 / mat_list[i+1][2]
          Ci = mat_list[i][3] * mat_list[i][4] * dx / self.delta_t + 1 / Re + 1 / Rw
          diag_w.append(-1 / Rw)
          diag_c.append(Ci)
          diag_e.append(-1 / Re)
          self.tau.append(mat_list[i][3] * mat_list[i][4] * dx / self.delta_t)
          lastl = False
          ## loop check
          #print('!!inside b22')
          #print('help: ', help)
          #print(help, mat_list[i][1])
          ## mesh check
          #print(dx)
          mesh.append(dx)
          continue

        Re = delta_x / mat_list[i][2]
        Rw = Re
        Ci = mat_list[i][3] * mat_list[i][4] * delta_x / self.delta_t + 1 / Re + 1 / Rw
        diag_w.append(-1 / Rw)
        diag_c.append(Ci)
        diag_e.append(-1 / Re)
        self.tau.append(mat_list[i][3] * mat_list[i][4] * delta_x / self.delta_t)
        ## mesh check
        #print(delta_x)
        mesh.append(delta_x)
      
    #self.R = diags([diag_w, diag_c, diag_e], [-1, 0, 1]).toarray()
    diag_e.append(0)
    self.R = np.array([diag_w, diag_c, diag_e])
    self.tau = np.array(self.tau)
        
    return self.R, self.tau, [self.R_i, self.R_e], mesh

  def solve_he(self, R_mat, tau, R_bound, initial, indoor, outdoor):
    initial = np.array(initial)
    indoor = np.array(indoor)
    outdoor = np.array(outdoor)
    results = []
    end = len(indoor)
    perc = 0

    for i in range(end):
      initial_res = np.array(indoor[i])
      initial_res = np.append(initial_res, initial)
      initial_res = np.append(initial_res, outdoor[i])
      results.append(initial_res)
      tau2 = tau * initial
      tau2[0] = tau2[0] + indoor[i] / R_bound[0]
      tau2[-1] = tau2[-1] + outdoor[i] / R_bound[1]
      initial = linalg.solve_banded((1, 1), R_mat, tau2)

    return results

  def q_Q(self, temperatures, mesh):
    mat_list = np.copy(self.mat_list)
    
    q = []
    for i in range(len(temperatures)):
      R = mesh[0] / mat_list[0][2] + self.Rsi
      q.append((temperatures[i][1] - temperatures[i][0]) / R)
    q = np.array(q)
    Q = np.sum(q)

    return q, Q

class U_heat_flux:
  def __init__(self, layers, Rsi=0.13, Rse=0.04):
    self.layers = layers
    self.Rsi = Rsi
    self.Rse = Rse

  def uval(self):
    mat_list = []
    help = 0
    for i in range(len(self.layers)):
      min = help
      max = help + self.layers[i]['layer_length']
      conductivity = self.layers[i]['conductivity']
      mat_list.append([min, max, conductivity])

    R = self.Rsi

    for i in range(len(mat_list)):
      R += (mat_list[i][1] - mat_list[i][0]) / mat_list[i][2]

    R += self.Rse
    
    return 1 / R

  def q_U(self, U, indoor, outdoor):
    mat_list = []
    help = 0
    for i in range(len(self.layers)):
      min = help
      max = help + self.layers[i]['layer_length']
      conductivity = self.layers[i]['conductivity']
      mat_list.append([min, max, conductivity])

    results = []
    point = 0
    points = [-0.02, point]

    for i in range(len(mat_list)):
      point += mat_list[i][1]
      points.append(point)
    
    points.append(point + 0.02)

    for i in range(len(indoor)):
      q = - U * (indoor[i] - outdoor[i])

      ith_result = [indoor[i]]
      next = indoor[i] + q * self.Rsi
      ith_result.append(next)
      
      for j in range(len(mat_list)):
        Ri = (mat_list[j][1] - mat_list[j][0]) / mat_list[j][2]
        next += q * Ri
        ith_result.append(next)
     
      ith_result.append(outdoor[i])
    
      results.append(ith_result)

    return results, points

  def q_Q(self, U, indoor, outdoor):
    q = []
    for i in range(len(indoor)):
      q.append(- U * (indoor[i] - outdoor[i]))
    q = np.array(q)
    Q = np.sum(q)

    return q, Q