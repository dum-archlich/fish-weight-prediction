import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
except ImportError:
    install("scikit-learn")
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np
    
try:
    file_csv = pd.read_csv('fishers_maket.csv')
except FileNotFoundError:
    print("Error: 'fishers_market.csv' file not found.")
    raise

print("--- 2. Analisis Atribut ---")
print("Contoh 5 baris pertama data:")
print(file_csv.head())
print("\nInfo tipe data:")
file_csv.info()

features = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
label = 'Weight'

X = file_csv[features]
y = file_csv[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print("Model berhasil dilatih.")
print(f"Intercept (titik potong): {model.intercept_:.4f}\n")

coeff_df = pd.DataFrame(model.coef_, features, columns=['Koefisien'])
print("Koefisien (faktor pengali) untuk setiap atribut:\n")
print(coeff_df)

print("\n\n--- 4. Hasil Evaluasi Model ---")
y_pred = model.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

print("\n--- Penjelasan Evaluasi ---")
print(f"R-squared (R²): {r2*100:.2f}%")
print(f"Artinya, model ini dapat menjelaskan sekitar {r2*100:.2f}% variasi berat ikan berdasarkan fitur-fitur ukurannya.")
print(f"MAE: {mae:.4f}")
print(f"Artinya, secara rata-rata, prediksi model meleset sekitar {mae:.4f} gram dari berat ikan yang sebenarnya (pada data uji).")