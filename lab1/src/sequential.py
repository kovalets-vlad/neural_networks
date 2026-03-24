import pandas as pd
import matplotlib.pyplot as plt
import os  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping

ACTIVATIONS = ['relu', 'tanh', 'sigmoid']
OPTIMIZERS = ['Adam', 'RMSprop', 'SGD', 'Nadam', 'Adamax']

def plot_comparison(data_dict, title, ylabel, is_log=False, save_path=None):
    plt.figure(figsize=(10, 5))
    for name, history in data_dict.items():
        plt.plot(history, label=name)
    plt.title(title)
    plt.xlabel('Епохи')
    plt.ylabel(ylabel)
    if is_log: plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Graph saved to: {save_path}")
    
    plt.show()
    plt.close()

def build_model(input_dim, task_type, opt='adam', act='relu'):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    if task_type == 'regression':
        model.add(Dense(64, activation=act))
        model.add(Dense(32, activation=act))
        model.add(Dense(1) )
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    else:
        model.add(Dense(32, activation=act))
        model.add(Dense(16, activation=act))
        model.add(Dense(1, activation='sigmoid')) 
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
    return model

def run_experiment(file_path, target_col, task_type, folder_name):
    print(f"\n{'='*20} Running {task_type.upper()}: {file_path} {'='*20}")

    base_results_path = f"results/{folder_name}"
    prefix = "reg" if task_type == "regression" else "class"

    df = pd.read_csv(file_path)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    input_dim = X_train.shape[1]
    metric_name = 'MSE' if task_type == 'regression' else 'Accuracy'
    metric_key = 'val_loss' if task_type == 'regression' else 'val_accuracy'
    
    opt_results = {}
    for opt in OPTIMIZERS:
        model = build_model(input_dim, task_type, opt=opt)
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        h = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, verbose=0, callbacks=[es])
        opt_results[opt] = h.history[metric_key]
    
    plot_comparison(
        opt_results, 
        f'{task_type}: Optimizers', 
        metric_name, 
        is_log=(task_type=='regression'),
        save_path=f"{base_results_path}/{prefix}_optimizers.png"
    )

    act_results = {}
    for act in ACTIVATIONS:
        model = build_model(input_dim, task_type, opt='adam', act=act)
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        h = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, verbose=0, callbacks=[es])
        act_results[act] = h.history[metric_key]
        
    plot_comparison(
        act_results, 
        f'{task_type}: Activations', 
        metric_name, 
        is_log=(task_type=='regression'),
        save_path=f"{base_results_path}/{prefix}_activations.png"
    )

    print(f"\nFinal Evaluation ({task_type}):")
    final_model = build_model(input_dim, task_type, opt='adam', act='relu')
    es_final = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    final_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, verbose=0, callbacks=[es_final])

    res_train = final_model.evaluate(X_train, y_train, verbose=0)
    res_test = final_model.evaluate(X_test, y_test, verbose=0)

    if task_type == 'regression':
        print(f"Train MSE: {res_train[0]:.4f}, MAE: {res_train[1]:.4f}")
        print(f"Test  MSE: {res_test[0]:.4f}, MAE: {res_test[1]:.4f}")
    else:
        print(f"Train Loss: {res_train[0]:.4f}, Acc: {res_train[1]:.4f}")
        print(f"Test  Loss: {res_test[0]:.4f}, Acc: {res_test[1]:.4f}")

run_experiment("data/hourly_wages_data.csv", "wage_per_hour", "regression", "hourly_wages")
run_experiment("data/diabetes_data.csv", "diabetes", "classification", "diabetes")
run_experiment("data/winequality-red.csv", "quality", "regression", "wine_quality")