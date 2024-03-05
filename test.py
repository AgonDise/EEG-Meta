import numpy as np
import matplotlib.pyplot as plt
file_path ='EEG_Spectrograms/568657.npy'
data = np.load(file_path)
print(data)
time = np.arange(len(data))

spectrogram_data = np.load(file_path)

plt.figure(figsize=(10, 5))
plt.imshow(spectrogram_data, aspect='auto', origin='lower', cmap='jet')
plt.title("Spectrogram của EEG")
plt.xlabel("Thời gian")
plt.ylabel("Tần số")
plt.colorbar(label='Cường độ')
plt.show()