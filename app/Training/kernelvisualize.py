import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Contoh data untuk SVM Linear
X_linear = np.array([[1, 2], [2, 3], [2, 4], [3, 3], [6, 5], [7, 7], [8, 6], [7, 8]])
y_linear = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Contoh data untuk SVM Non-Linear
X_non_linear = np.array([[1, 2], [2, 3], [2, 4], [3, 3], [3, 1], [4, 2], [5, 1], [4, 4]])
y_non_linear = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Buat model SVM Linear
clf_linear = svm.SVC(kernel='rbf')
clf_linear.fit(X_linear, y_linear)

# Buat model SVM Non-Linear dengan kernel polinomial
clf_non_linear = svm.SVC(kernel='poly', degree=2)
clf_non_linear.fit(X_non_linear, y_non_linear)

# Fungsi untuk membuat plot hasil klasifikasi
def plot_decision_boundary(clf, X, y, title):
    # Buat meshgrid untuk menggambar garis pemisah
    h = 0.02  # Step size pada mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Prediksi nilai kelas untuk setiap titik pada meshgrid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot hasil klasifikasi
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    
    # Plot data asli
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    
    plt.xlabel('Fitur 1')
    plt.ylabel('Fitur 2')
    plt.title(title)
    plt.show()

# Visualisasi hasil klasifikasi SVM Linear
plot_decision_boundary(clf_linear, X_linear, y_linear, 'SVM Linear')

# Visualisasi hasil klasifikasi SVM Non-Linear dengan kernel polinomial
plot_decision_boundary(clf_non_linear, X_non_linear, y_non_linear, 'SVM Non-Linear (Kernel RBF)')
