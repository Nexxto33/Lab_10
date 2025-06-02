# Importar librerías
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Cargar el dataset y seleccionar solo columnas necesarias
df = pd.read_csv('/content/spam text data.csv', encoding='latin-1')[['Category', 'Message']]
df.columns = ['label', 'text']  # renombrar columnas para simplicidad
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # convertir etiquetas a 0/1

# Tokenización del texto
tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, maxlen=50, padding='post')

# Separar en entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(padded, df['label'], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=50),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar y entrenar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluación final
print("Evaluación en el set de prueba:")
model.evaluate(X_test, y_test)
