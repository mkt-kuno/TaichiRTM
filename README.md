# Reverse-Time-Migration
We have created a program for reverse time migration in seismic exploration using a two-dimensional elastic wave propagation model.　

# Characteristics
## Reverse time migration program for seismic exploration
Reverse Time Migration(RTM) is a seismic wave inversion method that calculates reflection cross sections by combining (closs-correlation, convolution, etc...) seismic wave forward and backward propagation data by the observed waveform and source.
This program implements RTM using seismic wave forward propagation modeling and backward propagation equation of P-SV wave and SH wave.
## Python + GPU (CuPy) 
While based on Python, parallel processing using a GPU is possible by using CuPy.

# Examples 
Sample waveform file in np.array format. Analysis procedure was shown in example.py. 
example rho model　　　　　　　　　　calculated closs-section of example  
<img src="https://github.com/HaraandYutaro/Reverse-Time-Migration/blob/main/examples/ex%20model/Ex_rhomodel.png" width="300" alt="Sample Image" /> <img src='https://github.com/HaraandYutaro/Reverse-Time-Migration/blob/main/examples/results/RTMimages/y_120.png' width="400" alt="Sample Image" />

# How to use
## 1. Input data
Prepare record waveforms in np.array format
## 2. Parameter settings
Set parameters such as the size of the analysis area, sampling rate, wave velocity, and source location (See example.py)
## 3. RTM execution
Call the core functions of RTM using the CuPy backend
The wave is propagated forward and backward, and the imaging results are generated according to the imaging conditions (e.g., cross-correlation).
The result files are output in npz format
## 4. Visualization of the results
Load the npz file of the results, apply stacking and surface noise removal as necessary, and draw the reflection cross-section.
imaging.py uses matplotlib and other tools to output images


