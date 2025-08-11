import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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

def load_model_and_preprocessors():
    """Load the saved model and recreate preprocessors"""
    # Load the saved model
    model = IrisCNN()
    model.load_state_dict(torch.load('best_cnn_model.pth', map_location='cpu'))
    model.eval()
    
    # Load the original data to recreate preprocessors
    data = pd.read_csv('IRIS.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Recreate scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Recreate label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    return model, scaler, label_encoder

def predict_iris(model, scaler, label_encoder, sample):
    """Predict iris species for a new sample"""
    # Preprocess sample
    sample_scaled = scaler.transform([sample])
    sample_tensor = torch.FloatTensor(sample_scaled).unsqueeze(1)  # Add channel dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(sample_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_prob = torch.max(probabilities).item()
        all_probs = probabilities.numpy()[0]
    
    predicted_species = label_encoder.classes_[predicted_class]
    
    return predicted_species, predicted_prob, all_probs

def main():
    print("=== Testing Saved Iris CNN Model ===\n")
    
    try:
        # Load model and preprocessors
        print("Loading saved model...")
        model, scaler, label_encoder = load_model_and_preprocessors()
        print("Model loaded successfully!")
        
        # Test samples
        test_samples = [
            [5.1, 3.5, 1.4, 0.2],  # Iris-setosa
            [6.3, 3.3, 4.7, 1.6],  # Iris-versicolor  
            [6.3, 3.3, 6.0, 2.5],  # Iris-virginica
            [5.8, 2.7, 4.1, 1.0],  # Iris-versicolor
            [7.0, 3.2, 4.7, 1.4],  # Iris-versicolor
            [6.5, 3.0, 5.2, 2.0],  # Iris-virginica
        ]
        
        print(f"\nTesting {len(test_samples)} samples...")
        print("=" * 60)
        
        for i, sample in enumerate(test_samples, 1):
            species, confidence, all_probs = predict_iris(model, scaler, label_encoder, sample)
            
            print(f"\nSample {i}: {sample}")
            print(f"Predicted: {species}")
            print(f"Confidence: {confidence:.4f}")
            print("Class probabilities:")
            for class_name, prob in zip(label_encoder.classes_, all_probs):
                print(f"  {class_name}: {prob:.4f}")
        
        print("\n" + "=" * 60)
        print("Model testing completed successfully!")
        
    except FileNotFoundError:
        print("Error: Model file 'best_cnn_model.pth' not found!")
        print("Please run the main training script first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 