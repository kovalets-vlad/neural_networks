import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import warnings 
import os
import seaborn as sns
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix 
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, RandomRotation, RandomZoom, RandomTranslation
from keras.optimizers import Adam, RMSprop, SGD, Nadam, Adamax 
from keras.models import Sequential
from keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore') 

BASE_RESULTS_DIR = "neural_networks/lab2/results/"
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

OPTIMIZERS_LIST = ['Adam', 'RMSprop', 'SGD', 'Nadam', 'Adamax']

train_path = os.path.join("neural_networks/lab2/input", "train.csv")
test_path = os.path.join("neural_networks/lab2/input", "test.csv")

if os.path.exists(train_path) and os.path.exists(test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    Y_train = train["label"] 
    X_train = train.drop(labels=["label"], axis=1) 
    X_train = X_train.values.reshape(-1, 28, 28, 1) / 255.0
    Y_train = to_categorical(Y_train, num_classes=10)
    X_test = test.values.reshape(-1, 28, 28, 1) / 255.0
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)
else:
    print("Помилка: Файли даних не знайдено в папці /input")

def build_model(x_size_1=5, y_size_1=5, x_size_2=3, y_size_2=3, add_extra_layer=False, aug_factor=0.05):
    layers = [
        Input(shape=(28, 28, 1)),
        RandomRotation(factor=aug_factor),
        RandomZoom(height_factor=aug_factor, width_factor=aug_factor), 
        RandomTranslation(height_factor=aug_factor, width_factor=aug_factor), 
        
        Conv2D(filters=8, kernel_size=(x_size_1, y_size_1), padding='Same', activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(filters=16, kernel_size=(x_size_2, y_size_2), padding='Same', activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25)
    ]

    if add_extra_layer:
        layers.extend([
            Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Dropout(0.25)
        ])
    
    layers.extend([
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])
    
    return Sequential(layers)

def plot_comparison(data_dict, title, ylabel, folder_name, filename):
    plt.figure(figsize=(12, 6))
    for name, values in data_dict.items():
        plt.plot(values, label=name, marker='o', markersize=3)
    plt.title(title, fontsize=14)
    plt.xlabel('Епохи', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_dir = os.path.join(BASE_RESULTS_DIR, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300)
    plt.show()
    plt.close()

def model_training(data_dict, test_key, folder_name, x_size_1=5, x_size_2=3, opt_name="Adam", add_extra_layer=False, aug_factor=0.05):
    print(f"\n{'='*20} Training: {test_key} | Opt: {opt_name} {'='*20}")
    
    model = build_model(x_size_1=x_size_1, x_size_2=x_size_2, add_extra_layer=add_extra_layer, aug_factor=aug_factor)
    
    optimizers_map = {
        'Adam': Adam(learning_rate=0.001),
        'RMSprop': RMSprop(learning_rate=0.001),
        'SGD': SGD(learning_rate=0.01),
        'Nadam': Nadam(learning_rate=0.001),
        'Adamax': Adamax(learning_rate=0.001)
    }
    
    model.compile(
        optimizer=optimizers_map[opt_name], 
        loss="categorical_crossentropy", 
        metrics=["accuracy"],
    )
    
    history = model.fit(
        x=X_train, 
        y=Y_train, 
        batch_size=250,
        epochs=25, 
        validation_data=(X_val, Y_val),
        verbose=0,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)]
    )

    data_dict[test_key] = history.history["val_accuracy"]

    predictions_val = model.predict(X_val)
    predicted_labels_val = np.argmax(predictions_val, axis=1)
    true_labels_val = np.argmax(Y_val, axis=1)
    cm = confusion_matrix(true_labels_val, predicted_labels_val)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {test_key}")
    plt.ylabel('Реальні класи')
    plt.xlabel('Передбачені класи')
    
    save_dir = os.path.join(BASE_RESULTS_DIR, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    safe_key = test_key.replace(" ", "_").replace(":", "")
    plt.savefig(os.path.join(save_dir, f"cm_{safe_key}.png"), dpi=300)
    plt.show()
    plt.close()

    return model

opt_dict = {}
for name in OPTIMIZERS_LIST:
    model_training(opt_dict, test_key=name, folder_name="optimizers", opt_name=name)
plot_comparison(opt_dict, "Accuracy Comparison: Optimizers", "Validation Accuracy", "optimizers", "summary_optimizers")

core_dict = {}
for size in [3, 5, 7]:
    key = f"Kernel size {size}x{size}"
    model_training(core_dict, test_key=key, folder_name="kernels", x_size_1=size, x_size_2=size)
plot_comparison(core_dict, "Accuracy Comparison: Kernel size", "Validation Accuracy", "kernels", "summary_kernels")

extra_dict = {}
for extra in [True, False]:
    key = "Deep (3 layers)" if extra else "Base (2 layers)"
    model_training(extra_dict, test_key=key, folder_name="depth", add_extra_layer=extra)
plot_comparison(extra_dict, "Accuracy Comparison: Model Depth", "Validation Accuracy", "depth", "summary_depth")

aug_dict = {}
for factor in [0.05, 0.1, 0.2]:
    key = f"Aug factor {factor}"
    model_training(aug_dict, test_key=key, folder_name="augmentation", aug_factor=factor)
plot_comparison(aug_dict, "Accuracy Comparison: Augmentation Factors", "Validation Accuracy", "augmentation", "summary_augmentation")