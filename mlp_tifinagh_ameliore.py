import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

# === Fonctions d'activation ===

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# === Augmentation de données ===

def augment_image(img):
    rows, cols = img.shape
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.uniform(-15, 15), 1)
    M_trans = np.float32([[1, 0, np.random.uniform(-3, 3)], [0, 1, np.random.uniform(-3, 3)]])
    img = cv2.warpAffine(img, M_rot, (cols, rows))
    img = cv2.warpAffine(img, M_trans, (cols, rows))
    return img
# === Validation croisée K-Fold ===

from sklearn.model_selection import StratifiedKFold

def run_k_fold_cross_validation(X, y, layer_sizes, n_splits=5, epochs=100, batch_size=32):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_scores = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        onehot = OneHotEncoder(sparse_output=False)
        y_train_oh = onehot.fit_transform(y_train.reshape(-1, 1))
        y_val_oh = onehot.transform(y_val.reshape(-1, 1))

        model = MultiClassNeuralNetwork(layer_sizes)
        _, _, train_accs, val_accs = model.train(X_train, y_train_oh, X_val, y_val_oh,
                                                 epochs=epochs, batch_size=batch_size)

        final_val_pred = model.forward(X_val)
        final_val_acc = model.compute_accuracy(y_val_oh, final_val_pred)
        print(f"Fold {fold + 1} Validation Accuracy: {final_val_acc:.4f}")
        all_scores.append(final_val_acc)

    print(f"\n=== K-Fold Cross-Validation Results ===")
    print(f"Mean Accuracy: {np.mean(all_scores):.4f}, Std: {np.std(all_scores):.4f}")
    return all_scores
# === Classe MLP avec Adam et L2 ===

class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.001):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
        # Adam
        self.v_dW = [np.zeros_like(w) for w in self.weights]
        self.s_dW = [np.zeros_like(w) for w in self.weights]
        self.v_db = [np.zeros_like(b) for b in self.biases]
        self.s_db = [np.zeros_like(b) for b in self.biases]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 1

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            self.activations.append(relu(z))
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = softmax(z)
        self.activations.append(output)
        return output

    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def compute_accuracy(self, y_true, y_pred):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

    def backward(self, y, outputs):
        m = y.shape[0]
        self.d_weights = [None] * len(self.weights)
        self.d_biases = [None] * len(self.biases)
        reg_lambda = 0.001
        dZ = outputs - y
        self.d_weights[-1] = (self.activations[-2].T @ dZ) / m + (reg_lambda / m) * self.weights[-1]
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m
        for i in range(len(self.weights) - 2, -1, -1):
            dZ = (dZ @ self.weights[i+1].T) * relu_derivative(self.z_values[i])
            self.d_weights[i] = (self.activations[i].T @ dZ) / m + (reg_lambda / m) * self.weights[i]
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m

    def update_weights_adam(self):
        for i in range(len(self.weights)):
            self.v_dW[i] = self.beta1 * self.v_dW[i] + (1 - self.beta1) * self.d_weights[i]
            self.s_dW[i] = self.beta2 * self.s_dW[i] + (1 - self.beta2) * (self.d_weights[i] ** 2)
            v_dW_corr = self.v_dW[i] / (1 - self.beta1 ** self.t)
            s_dW_corr = self.s_dW[i] / (1 - self.beta2 ** self.t)

            self.v_db[i] = self.beta1 * self.v_db[i] + (1 - self.beta1) * self.d_biases[i]
            self.s_db[i] = self.beta2 * self.s_db[i] + (1 - self.beta2) * (self.d_biases[i] ** 2)
            v_db_corr = self.v_db[i] / (1 - self.beta1 ** self.t)
            s_db_corr = self.s_db[i] / (1 - self.beta2 ** self.t)

            self.weights[i] -= self.learning_rate * v_dW_corr / (np.sqrt(s_dW_corr) + self.epsilon)
            self.biases[i]  -= self.learning_rate * v_db_corr / (np.sqrt(s_db_corr) + self.epsilon)
        self.t += 1

    def train(self, X, y, X_val, y_val, epochs=100, batch_size=32):
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                outputs = self.forward(X_batch)
                self.backward(y_batch, outputs)
                self.update_weights_adam()
            train_pred = self.forward(X)
            val_pred = self.forward(X_val)
            train_losses.append(self.compute_loss(y, train_pred))
            val_losses.append(self.compute_loss(y_val, val_pred))
            train_accs.append(self.compute_accuracy(y, train_pred))
            val_accs.append(self.compute_accuracy(y_val, val_pred))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Train Acc={train_accs[-1]:.4f}, Val Acc={val_accs[-1]:.4f}")
        return train_losses, val_losses, train_accs, val_accs

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# === Chargement des données ===

def load_and_preprocess_image(path, target_size=(32, 32), augment=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    if augment:
        img = augment_image(img)
    img = img.astype(np.float32) / 255.0
    return img.flatten()

data_dir = os.path.join(os.getcwd(), 'amhcd-data-64/tifinagh-images/')
image_paths, labels = [], []
for label_dir in os.listdir(data_dir):
    for file in os.listdir(os.path.join(data_dir, label_dir)):
        image_paths.append(os.path.join(label_dir, file))
        labels.append(label_dir)

df = pd.DataFrame({'image_path': image_paths, 'label': labels})
label_encoder = LabelEncoder()
df['encoded'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)

X = np.array([
    load_and_preprocess_image(os.path.join(data_dir, p), augment=True)
    for p in df['image_path']
])
y = df['encoded'].values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

onehot = OneHotEncoder(sparse_output=False)
y_train_oh = onehot.fit_transform(y_train.reshape(-1, 1))
y_val_oh = onehot.transform(y_val.reshape(-1, 1))
y_test_oh = onehot.transform(y_test.reshape(-1, 1))

# === Entraînement ===

model = MultiClassNeuralNetwork([1024,128, 64, 32, num_classes])
train_losses, val_losses, train_accs, val_accs = model.train(X_train, y_train_oh, X_val, y_val_oh)

# === Évaluation ===

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")

plt.savefig("metrics_curves.png")
plt.show()
