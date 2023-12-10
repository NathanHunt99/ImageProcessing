import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack 



x = np.matrix([[2,0,-2,0], [0,-2,0,2], [2,0,-2,0]])

#Image (x) is 3x4 grid (3 rows/4 columns)
#DFT of image has the same resolution (as in the 1D case where # bins = # samples)
#Image 3x4 so DFT 3x4
N_rows = 3
N_cols = 4
X_dft_rows = np.zeros((N_rows,N_cols), dtype=np.complex128) 
X_dft = np.zeros((N_rows,N_cols), dtype=np.complex128)

#First we take the DFT of each row and replace that row with its DFT
#So the result will be a new image of the DFT of each row where each pixel in the row
#is a new freq bin (so far left columns will be the 0 frequency bn for each rows DFT)

for r in range(N_rows):
    for k in range(N_cols):
        X_dft_rows[r,k] = np.sum(np.dot(x[r,:],np.exp(-2j*np.pi*(k/N_cols)*np.arange(N_cols))))

for c in range(N_cols):
    for k in range(N_rows):
            X_dft[k,c]= np.sum(np.dot(X_dft_rows[:,c], np.exp(-2j*np.pi*(k/N_rows)*np.arange(N_rows))))


X = fftpack.fft2(x) #Library Version

plt.imshow(x)
plt.xticks([-0.5, 0.5, 1.5, 2.5, 3.5])
plt.yticks([-0.5, 0.5, 1.5, 2.5])
plt.grid()
plt.title("Original Image")
plt.show()

plt.imshow(abs(X))
plt.xticks([-0.5, 0.5, 1.5, 2.5, 3.5])
plt.yticks([-0.5, 0.5, 1.5, 2.5])
plt.grid()
plt.title("MAG FFT of Image")
plt.show()

plt.imshow(abs(X_dft))
plt.xticks([-0.5, 0.5, 1.5, 2.5, 3.5])
plt.yticks([-0.5, 0.5, 1.5, 2.5])
plt.grid()
plt.title("MAG manual DFT of Image")
plt.show()
