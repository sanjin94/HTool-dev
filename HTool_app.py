import sys
sys.path.append('icons')
sys.path.append('data')
sys.path.append('calc')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import os
from PIL import Image


import htool as ht
import input_processing as inp

import streamlit as st
from datetime import datetime

EMOJI_ICON = "icons/HTool.ico"
EMOJI_PNG = "icons/HTool.png"


st.set_page_config(page_title="HTool", layout='wide')
col1, col2 = st.columns([2.5, 4])
with col2:
    st.image(EMOJI_PNG, width=200)
col1, col2 = st.columns([1,3])
with col2:
    st.title('HTool - 1D heat transfer tool')

# Sidebar
st.sidebar.header('Defining the calculation input')

bc = st.sidebar.radio('Select boundary condition type:', ['TRSYS measurement', 'Climate data for selected city'], key=None)

if bc == 'TRSYS measurement':
    uploaded_file = st.sidebar.file_uploader(
            'Upload a file to input data', type='dat')
    if uploaded_file is not None:
        with open('data/raw/' + uploaded_file.name, "wb") as f:
            os.path.relpath('/data/raw')
            f.write(uploaded_file.getbuffer())
    filenames = os.listdir('data/raw/')
    file_name = st.sidebar.selectbox('Select input file', filenames)
    start_date = st.sidebar.date_input('Select start date: ')
    start_time = st.sidebar.text_input('Write down start time (hh:mm):')
    end_date = st.sidebar.date_input('Select end date:')
    end_time = st.sidebar.text_input('Write down end time (hh:mm):')
elif bc == 'Climate data for selected city':
    st.sidebar.write('TODO')
    
initial = st.sidebar.radio('Please select initial temperature type:', ['Constant temperature', 'Steady-state transfer'])

if initial == 'Constant temperature':
    ctemp = st.sidebar.number_input('Write down constant initial temperature')
elif initial == 'Steady-state transfer':
    initial_indoor = st.sidebar.number_input('Write down indoor temperature')
    initial_outdoor = st.sidebar.number_input('Write down outdoor temperature')

# Main part
n_layers = 0
n_layers = st.number_input('Define number of layers', min_value = 0, max_value=10, value=0)

name = []
cond = []
rho = []
c = []
l = []

if n_layers != 0:
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    for i in range(n_layers):
        with col1:
            name.append(st.text_input(label=f'Layer {i+1} name', key=f'Question {i}'))
        with col2:
            cond.append(st.number_input(label=f'Layer {i+1} conductivity', key=f'Question {i}', step=1e-3, format="%.3f"))
        with col3:
            rho.append(st.number_input(label=f'Layer {i+1} density', key=f'Question {i}'))
        with col4:
            c.append(st.number_input(label=f'Layer {i+1} capacity', key=f'Question {i}'))
        with col5:
            l.append(st.number_input(label=f'Layer {i+1} length [m]', key=f'Question {i}', step=1e-3, format="%.3f"))

col1, col2 = st.columns([2.5,3])
with col2:
   initial_b = st.button('Calculate')        

if  initial_b:
    materials = ht.Material()
    for i in range(len(name)):
        materials.new_material(name[i], cond[i], rho[i], c[i], l[i])
    resistance = ht.Resistance(materials.layers, delta_x=0.005, delta_t = 60)
    R_mat, tau, R_bound, mesh = resistance.resistance_tau()

    if bc == 'TRSYS measurement':
        filename = 'data/raw/' + file_name
        start = str(start_date) + ' ' + str(start_time)
        end = str(end_date) + ' ' + str(end_time)
        index = inp.RawData_index(filename, start, end)
        first, last = index.dfinit()
        columns = index.cols()
        series = inp.RawData_series(filename, first, last, columns)
        vectors = series.ex_vect()
        indoor = vectors[0]
        outdoor = vectors[1]

        if initial == 'Constant temperature':
            initial = np.array([ctemp for i in range(len(tau))])
        elif initial == 'Steady-state transfer':
            middle = (initial_indoor + initial_outdoor) / 2
            help = np.array([middle for i in range(len(tau))])
            outdoor_init = np.array([initial_outdoor for i in range(60*60*3)])
            indoor_init = np.array([initial_indoor for i in range(60*60*3)])
            res_init = resistance.solve_he(R_mat, tau, R_bound, help, indoor_init, outdoor_init)
            initial = res_init[-1][1:len(res_init[-1])-1]
        
        try:
            initial = initial
            plt.plot(initial) #TODO insert legend
            plt.savefig('results/first_plot/initial')
            plt.clf()
            plt.plot(indoor, label='indoor temperature') #TODO insert legend
            plt.plot(outdoor, label='outdoor temperature') #TODO insert legend
            plt.savefig('results/first_plot/bc')
            plt.clf()
            col1, col2 = st.columns([3,3])
            with col1:
                initial_f = Image.open('results/first_plot/initial.png')
                st.image(initial_f, caption='Initial condition')
            with col2:
                bc_f = Image.open('results/first_plot/bc.png')
                st.image(bc_f, caption='Boundary conditions')
        except:
            pass

    elif bc == 'Climate data for selected city':
        st.warning('TODO')


    results = resistance.solve_he(R_mat, tau, R_bound, initial, indoor, outdoor)
    q_calc, Q_calc = resistance.q_Q(results, mesh)
    plt.plot(q_calc, label="q_transient")
    U_cls = ht.U_heat_flux(materials.layers)
    U_val = U_cls.uval()
    U_results, points = U_cls.q_U(U_val, indoor, outdoor)
    q_U, Q_U = U_cls.q_Q(U_val, indoor, outdoor)
    plt.plot(q_U, label='q_steady-state')
    plt.savefig('results/q.png')
    plt.clf()

    mesh2 = np.array([-0.01])
    mesh2 = np.append(mesh2, mesh)
    mesh2 = np.append(mesh2, mesh2[-1]+0.01)
    

    min_bc = np.amin(indoor)
    min2 = np.amin(outdoor)
    if min2 < min_bc:
        min_bc = min2
    
    max_bc = np.amax(indoor)
    max2 = np.amax(outdoor)
    if max2 > max_bc:
        max_bc = max2

    col1, col2 = st.columns([3,3])
    with col1:
        imageLocation1 = st.empty()
    with col2:
        imageLocation2 = st.empty()

    for i in range(len(outdoor)):
        if i % 60 == 0:
            plt.ylim(min_bc, max_bc)
            plt.plot(results[i])
            plt.savefig('results/transient/last')
            plt.clf()
            transient = Image.open('results/transient/last.png')
            imageLocation1.image(transient, caption='Transient')
            plt.ylim(min_bc, max_bc)
            plt.plot(points, U_results[i])
            plt.savefig('results/stationary/last')
            plt.clf()
            stationary = Image.open('results/stationary/last.png')
            imageLocation2.image(stationary, caption='Steady-state')
    
    col1, col2 = st.columns([3,3])
    with col1:
        q_fig = Image.open('results/q.png')
        st.image(q_fig, caption='Comparison between q_transient and q_steady state')
    with col2:
        pass #TODO dodaj tablice

    



    

