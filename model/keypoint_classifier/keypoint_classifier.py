import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib

# define a MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class KeyPointClassifier:
    def __init__(self,
                 label_csv='keypoint_classifier_label.csv',
                 data_csv='keypoint.csv',
                 model_path='pytorch_mlp_model.pth',
                 scaler_path='scaler.save',
                 batch_size=32,
                 learning_rate=0.0001,
                 num_epochs=20,
                 hidden_dim=100,
                 test_size=0.3,
                 random_state=42,
                 load_existing_model=True):  

        # Get the absolute directory of the current script
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path of the files
        label_csv_path = os.path.join(base_dir, label_csv)
        data_csv_path = os.path.join(base_dir, data_csv)
        self.model_path = os.path.join(base_dir, model_path)
        self.scaler_path = os.path.join(base_dir, scaler_path)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_dim = hidden_dim
        self.test_size = test_size
        self.random_state = random_state

        # Read the label file and create a mapping
        label_mapping = pd.read_csv(label_csv_path, header=None)
        self.label_dict = {i: label_mapping.iloc[i, 0] for i in range(len(label_mapping))}
        self.labels = self.label_dict

        # Read training data from keypoint.csv
        data = pd.read_csv(data_csv_path, header=0)
        self.y = data.iloc[:, 0].values  
        self.X = data.iloc[:, 1:].values  

        #  Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # Standardization
        if os.path.exists(self.scaler_path) and load_existing_model:
            print(f"Loading existing scaler from {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            self.X_train = self.scaler.transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        else:
            print("No existing scaler found, creating a new one.")
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

        # Convert to PyTorch Tensors and construct DataLoader
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Construct the model
        self.input_dim = self.X_train.shape[1]  # features
        self.num_classes = len(self.label_dict)  # classes
        self.model = MLP(self.input_dim, self.hidden_dim, self.num_classes)

        # Define the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Load an existing model）
        if os.path.exists(self.model_path) and load_existing_model:
            print(f"Loading existing model from {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            print("No existing model found, training from scratch.")

    def train_model(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"Model saved to {self.model_path}")
        print(f"Scaler saved to {self.scaler_path}")

    def predict(self, landmark_list):
        import numpy as np
        landmark_array = np.array(landmark_list).reshape(1, -1)
        landmark_array = self.scaler.transform(landmark_array)
        input_tensor = torch.tensor(landmark_array, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()

    def __call__(self, landmark_list):
        return self.predict(landmark_list)


if __name__ == '__main__':
    classifier = KeyPointClassifier(load_existing_model=True) 
    classifier.train_model()  
    classifier.evaluate() 
    classifier.save_model()  
