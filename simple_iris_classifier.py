import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class IrisCNN(nn.Module):
    """1D CNN for iris classification"""
    def __init__(self, input_size=4, num_classes=3):
        super(IrisCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=2, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(16, 32, kernel_size=2, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        conv_output_size = 32 * 2
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def train_and_evaluate():
    """Train the CNN model and return accuracies"""
    print("=== Training Iris CNN Classifier ===\n")
    
    # Load data
    data = pd.read_csv('IRIS.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create and train model
    model = IrisCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training model...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")
    
    # Evaluate on training set
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        _, train_pred = torch.max(train_outputs, 1)
        train_accuracy = accuracy_score(y_train, train_pred.numpy())
        
        test_outputs = model(X_test_tensor)
        _, test_pred = torch.max(test_outputs, 1)
        test_accuracy = accuracy_score(y_test, test_pred.numpy())
    
    print(f"\n=== Results ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print(f"\n=== Classification Report ===")
    print(classification_report(y_test, test_pred.numpy(), target_names=label_encoder.classes_))
    
    return model, scaler, label_encoder

def predict_user_input(model, scaler, label_encoder):
    """Get user input and predict iris class"""
    print(f"\n=== Predict New Iris Sample ===")
    print("Enter the 4 features (sepal_length, sepal_width, petal_length, petal_width):")
    
    try:
        sepal_length = float(input("Sepal Length: "))
        sepal_width = float(input("Sepal Width: "))
        petal_length = float(input("Petal Length: "))
        petal_width = float(input("Petal Width: "))
        
        sample = [sepal_length, sepal_width, petal_length, petal_width]
        
        # Preprocess sample
        sample_scaled = scaler.transform([sample])
        sample_tensor = torch.FloatTensor(sample_scaled).unsqueeze(1)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(sample_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_prob = torch.max(probabilities).item()
            all_probs = probabilities.numpy()[0]
        
        predicted_species = label_encoder.classes_[predicted_class]
        
        print(f"\nPrediction Results:")
        print(f"Input features: {sample}")
        print(f"Predicted Class: {predicted_species}")
        print(f"Confidence: {predicted_prob:.4f}")
        print(f"Class Probabilities:")
        for class_name, prob in zip(label_encoder.classes_, all_probs):
            print(f"  {class_name}: {prob:.4f}")
            
    except ValueError:
        print("Error: Please enter valid numbers!")
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("Iris Flower Classification using CNN")
    print("=" * 50)
    
    # Train and evaluate model
    model, scaler, label_encoder = train_and_evaluate()
    
    # Interactive prediction
    while True:
        print(f"\n" + "="*50)
        choice = input("Do you want to predict a new iris sample? (y/n): ").lower()
        
        if choice == 'y' or choice == 'yes':
            predict_user_input(model, scaler, label_encoder)
        elif choice == 'n' or choice == 'no':
            print("Thank you for using the Iris Classifier!")
            break
        else:
            print("Please enter 'y' or 'n'")

if __name__ == "__main__":
    main() 