import numpy as np
import matplotlib.pyplot as plt

# Contoh dataset
jenis_ras = ['Persia', 'Russian Blue', 'Siamese', 'Bengal', 'Ragdoll']
jumlah_kucing_train = [400, 400, 400, 400, 400]
jumlah_kucing_test = [50, 50, 50, 50, 50]

# Menentukan lebar bar
bar_width = 0.55

# Menghasilkan posisi bar untuk setiap jenis ras kucing
train_pos = np.arange(len(jenis_ras))
test_pos = train_pos + bar_width

# Visualisasi dengan grouped bar chart
plt.bar(train_pos, jumlah_kucing_train, bar_width, label='Train')
plt.bar(test_pos, jumlah_kucing_test, bar_width, label='Test')

plt.xlabel('Jenis Ras Kucing')
plt.ylabel('Jumlah Kucing')
plt.title('Distribusi Jenis Ras Kucing (Train & Test)')
plt.xticks(train_pos + bar_width/2, jenis_ras)
plt.legend()
plt.show()
