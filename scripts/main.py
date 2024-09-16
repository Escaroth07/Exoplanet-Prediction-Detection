import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import models as m
import io
import PIL.Image
import PIL.ImageTk
import joblib

class MLModelTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Model Trainer")
        
        self.train_data = None
        self.test_data = None
        self.model = None
        
        # File Upload Section
        self.create_file_upload_section()
        
        # Model Selection Section
        self.create_model_selection_section()
        
        # Training Section
        self.create_training_section()
        
        # Evaluation Section
        self.create_evaluation_section()
        
        # Model Management Section
        self.create_model_management_section()
    
    def create_file_upload_section(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Label(frame, text="Upload Training Data:").grid(row=0, column=0, padx=10)
        self.upload_train_btn = tk.Button(frame, text="Browse...", command=self.upload_train_data)
        self.upload_train_btn.grid(row=0, column=1, padx=10)
        
        tk.Label(frame, text="Upload Testing Data:").grid(row=1, column=0, padx=10)
        self.upload_test_btn = tk.Button(frame, text="Browse...", command=self.upload_test_data)
        self.upload_test_btn.grid(row=1, column=1, padx=10)
        
    def create_model_selection_section(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Label(frame, text="Select Model:").grid(row=0, column=0, padx=10)
        self.model_var = tk.StringVar(value="FCN")
        self.fcn_rb = tk.Radiobutton(frame, text="FCN", variable=self.model_var, value="FCN")
        self.fcn_rb.grid(row=0, column=1, padx=10)
        self.svc_rb = tk.Radiobutton(frame, text="SVC", variable=self.model_var, value="SVC")
        self.svc_rb.grid(row=0, column=2, padx=10)

    def create_training_section(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        self.train_btn = tk.Button(frame, text="Train Model", command=self.train_model)
        self.train_btn.pack()

    def create_evaluation_section(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        self.evaluate_btn = tk.Button(frame, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_btn.pack()

        self.results_text = tk.Text(frame, height=10, width=60)
        self.results_text.pack()

    def create_model_management_section(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        self.save_model_btn = tk.Button(frame, text="Save Model", command=self.save_model)
        self.save_model_btn.pack(side=tk.LEFT, padx=10)

        self.load_model_btn = tk.Button(frame, text="Load Model", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=10)

    def upload_train_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.train_data = pd.read_csv(file_path)
            messagebox.showinfo("Info", "Training data uploaded successfully.")

    def upload_test_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.test_data = pd.read_csv(file_path)
            messagebox.showinfo("Info", "Testing data uploaded successfully.")

    def train_model(self):
        if self.train_data is None or self.test_data is None:
            messagebox.showerror("Error", "Please upload both training and testing data.")
            return
        
        # Data preprocessing
        scaler = StandardScaler()
        data_train = np.asarray(self.train_data)
        data_test = np.asarray(self.test_data)
        
        y_train = data_train[:, 0]
        y_test = data_test[:, 0]
        X_train = data_train[:, 1:]
        X_test = data_test[:, 1:]
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        rm = RandomOverSampler(sampling_strategy=0.5)
        X_train_ovs, y_train_ovs = rm.fit_resample(X_train, y_train)
        
        len_seq = X_train_ovs.shape[1]
        X_train_ovs_nn = X_train_ovs.reshape((X_train_ovs.shape[0], X_train_ovs.shape[1], 1))
        X_test_nn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Create model based on selection
        if self.model_var.get() == "FCN":
            self.model = m.FCN_model(len_seq)
            self.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
            history = self.model.fit(X_train_ovs_nn, y_train_ovs, epochs=15, batch_size=10, validation_data=(X_test_nn, y_test))
            
        elif self.model_var.get() == "SVC":
            self.model = m.SVC_model(C=100, gamma=0.001, kernel='rbf')
            self.model.fit(X_train_ovs, y_train_ovs)

        messagebox.showinfo("Info", "Model trained successfully.")

    def evaluate_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model trained yet.")
            return

        if self.test_data is None:
            messagebox.showerror("Error", "No testing data loaded.")
            return
    
        # Convert test data to numpy array and check its shape
        data_test = np.asarray(self.test_data)
        print("Data Test Shape:", data_test.shape)  

        if data_test.ndim == 1:
            data_test = data_test.reshape(1, -1)  #
    
        y_test = data_test[:, 0]  
        X_test = data_test[:, 1:]

        X_test = StandardScaler().fit_transform(X_test)
        X_test_nn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Evaluate model based on selection
        if self.model_var.get() == "FCN":
            y_test_pred = self.model.predict(X_test_nn)
            y_test_pred = (y_test_pred > 0.5)
            accuracy = accuracy_score(y_test, y_test_pred)
            report = classification_report(y_test, y_test_pred, target_names=["NO exoplanet confirmed", "YES exoplanet confirmed"])
            conf_matrix = confusion_matrix(y_test.astype(int), y_test_pred.astype(int))
        
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.show()
        
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Accuracy: {accuracy:.4f}\n\n")
            self.results_text.insert(tk.END, f"Classification Report:\n{report}")

        elif self.model_var.get() == "SVC":
            y_pred_svc = self.model.predict(X_test)
            report = classification_report(y_test, y_pred_svc, target_names=["NO exoplanet confirmed", "YES exoplanet confirmed"])
            conf_matrix = confusion_matrix(y_test.astype(int), y_pred_svc.astype(int))
        
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.show()
        
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Classification Report:\n{report}")

    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model to save.")
            return

        # Check if it's an FCN (Keras model) or SVC (sklearn model)
        model_type = self.model_var.get()
        file_path = filedialog.asksaveasfilename(defaultextension=".h5" if model_type == "FCN" else ".pkl", 
                                             filetypes=[("HDF5 Files", "*.h5")] if model_type == "FCN" else [("Pickle Files", "*.pkl")])
    
        if file_path:
            if model_type == "FCN":
                self.model.save(file_path) 
            elif model_type == "SVC":
                joblib.dump(self.model, file_path)  # Save SVC model using joblib in .pkl format
                messagebox.showinfo("Info", f"{model_type} model saved successfully.")

    def load_model(self):
        model_type = self.model_var.get()
        file_path = filedialog.askopenfilename(filetypes=[("HDF5 Files", "*.h5")] if model_type == "FCN" else [("Pickle Files", "*.pkl")])
    
        if file_path:
            if model_type == "FCN":
                self.model = tf.keras.models.load_model(file_path)  
            elif model_type == "SVC":
                self.model = joblib.load(file_path)  
                messagebox.showinfo("Info", f"{model_type} model loaded successfully.")

        
# Create the main window and run the app
root = tk.Tk()
app = MLModelTrainerApp(root)
root.mainloop()
