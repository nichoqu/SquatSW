import streamlit as st
import numpy as np
import joblib
from  tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import   streamlit  as st; from PIL import Image; import numpy  as np
import pandas  as pd; import pickle

import os

filename1 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture1.PNG'
filename2 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture2.PNG'

st.title('Probabilistic Hybrid Machine Learning Models for Predicting Shear Strength of Flanged Squat Reinforced Concrete Wallss')
with st.container():
    st.image(filename1)
    st.image(filename2)


import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Arrange input boxes into three columns for input features
col1, col2, col3 = st.columns(3)

# First row of inputs
with col1:
    hw_lw = st.number_input('hw/lw', 0.25, 2.0, step=0.01)
with col2:
    A = st.number_input('A (m^2)', 0.01, 0.82, step=0.01)
with col3:
    fc = st.number_input('fc (MPa)', 13.8, 110.7, step=0.1)

# Second row of inputs
col4, col5, col6 = st.columns(3)
with col4:
    fyv = st.number_input('fyv (MPa)', 235.0, 792.0, step=1.0)
with col5:
    fyh = st.number_input('fyh (MPa)', 235.0, 792.0, step=1.0)
with col6:
    fyf = st.number_input('fyf (MPa)', 235.0, 792.0, step=1.0)

# Third row of inputs
col7, col8, col9 = st.columns(3)
with col7:
    pv = st.number_input('pv (%)', 0.17, 2.54, step=0.01)
with col8:
    ph = st.number_input('ph (%)', 0.28, 1.69, step=0.01)
with col9:
    pf = st.number_input('pf (%)', 0.35, 6.4, step=0.01)

# Fourth row of inputs
col10, col11, col12 = st.columns(3)
with col10:
    Sv = st.number_input('Sv (mm)', 12.5, 280.0, step=1.0)
with col11:
    Sh = st.number_input('Sh (mm)', 12.5, 300.0, step=1.0)
with col12:
    P_fc_A = st.number_input('P/fc.A (%)', 0.0, 42.35, step=0.01)

# Gather all inputs into a list for checking
input_values = [hw_lw, A, fc, fyv, fyh, fyf, pv, ph, pf, Sv, Sh, P_fc_A]

# Normalize inputs using the min-max scaling to [-1, 1]
def normalize(value, min_val, max_val):
    return (2 * (value - min_val) / (max_val - min_val)) - 1

normalized_inputs = [
    normalize(hw_lw, 0.25, 2.0),
    normalize(A, 0.01, 0.82),
    normalize(fc, 13.8, 110.7),
    normalize(fyv, 235.0, 792.0),
    normalize(fyh, 235.0, 792.0),
    normalize(fyf, 235.0, 792.0),
    normalize(pv, 0.17, 2.54),
    normalize(ph, 0.28, 1.69),
    normalize(pf, 0.35, 6.4),
    normalize(Sv, 12.5, 280.0),
    normalize(Sh, 12.5, 300.0),
    normalize(P_fc_A, 0.0, 42.35),
]

# Convert to numpy array for model input
inputvec = np.array(normalized_inputs).reshape(1, -1, 1)

# Check for zeros
zero_count = sum(1 for value in input_values if value == 0)

# Load models and predict the outputs when the button is pressed
if st.button('Run'):

    # Validation: If more than 3 inputs are zero, show a warning message
    if zero_count > 2:
        st.error(f"Error: More than two input values are zero. Please provide valid inputs for at least 11 features.")
    else:

        def loadList(filename):
            # the filename should mention the extension 'npy'
            tempNumpyArray=np.load(filename)
            return tempNumpyArray.tolist()


        def decode_solution(model, solution):
            # solution: is a vector.
            # solution = [w11, w21, w31, w12, w22, w32, b1, b2, wh11, wh21, wh12, wh22, wh13, wh23, bo1, bo2, bo3 ]
            # number of weights = n_inputs * n_hidden_nodes + n_hidden_nodes + n_hidden_nodes * n_outputs + n_outputs
            # we decode the solution into the neural network weights
            # we return the model with the new weight (weight from solution)
            weight_sizes = [(w.shape, np.size(w)) for w in model.get_weights()]
            # ( (3, 5),  15 )
            weights = []
            cut_point = 0
            for ws in weight_sizes:
                temp = np.reshape(solution[cut_point: cut_point + ws[1]], ws[0])
                # [0: 15], (3, 5),
                weights.append(temp)
                cut_point += ws[1]
            model.set_weights(weights)

            return model

        ## define model
        def   create_model(n_hidden_nodes, n_inputs):
            model = Sequential()
            model.add(Dense(n_hidden_nodes, input_dim=n_inputs, activation='relu'))
            model.add(Dense(1, activation='linear'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        # Load models
        n_inputs = 12
        j = 8
        bsolution=np.array(loadList('./SHADE.npy')[0])
        nmodel4 = create_model(j, n_inputs)
        model4 = decode_solution(nmodel4, bsolution)
        # Predict outputs
        yhat1 = model4.predict(inputvec)  



        # Convert predictions back to original scales
        min_val = 18
        max_val = 7060
        Output1_real = ((yhat1 + 1) / 2) * (max_val - min_val) + min_val

        # Display predictions
        col13, col14, col15 = st.columns(3)
        with col13:
            st.write("Seah strength is: ", np.round(abs(Output1_real), decimals=2))


filename7 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture3.PNG'
filename8 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture4.PNG'

col22, col23 = st.columns(2)
with col22:
    with st.container():
        st.markdown("<h5>Developer:</h5>", unsafe_allow_html=True)
        st.image(filename8)

with col23:
    with st.container():
        st.markdown("<h5>Supervisor:</h5>", unsafe_allow_html=True)
        st.image(filename7) 


footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
    This web app was developed in School of Resources and Safety Engineering, Central South University, Changsha 410083, China
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
