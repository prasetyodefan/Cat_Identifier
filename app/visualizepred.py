import matplotlib.pyplot as plt

# Contoh dataset
jenis_ras = ['Bengal', 'Siamese', 'Russian Blue', 'Persian', 'Ragdoll']
jumlah_kucing = [88, 90, 82, 86, 82]

# Visualisasi dengan bar plot
plt.bar(jenis_ras, jumlah_kucing)

# Menambahkan nilai di dalam bar plot
for i in range(len(jenis_ras)):
    plt.text(i, jumlah_kucing[i], str(jumlah_kucing[i]), ha='center', va='bottom')

plt.xlabel('Jenis Ras Kucing')
plt.ylabel('Akurasi Kebenaran %')
plt.title('Akurasi Kebenaran Prediksi 5 Jenis Ras Kucing')

plt.show()
