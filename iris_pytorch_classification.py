import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class IrisCNN(nn.Module):
    """1D CNN for iris classification"""
    def __init__(self, input_size=4, num_classes=3):
        super(IrisCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv1d(1, 16, kernel_size=2, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second conv layer
            nn.Conv1d(16, 32, kernel_size=2, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Calculate the output size after convolutions
        # Input: (batch, 1, 4)
        # After first conv: (batch, 16, 3) - kernel_size=2, no padding
        # After second conv: (batch, 32, 2) - kernel_size=2, no padding
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
        # x shape: (batch_size, 1, 4)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class IrisDense(nn.Module):
    """Dense neural network for iris classification"""
    def __init__(self, input_size=4, num_classes=3):
        super(IrisDense, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

class IrisClassifier:
    def __init__(self, device='cpu'):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = device
        self.models = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the iris dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
        self.data = pd.read_csv('IRIS.csv')
        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {list(self.data.columns[:-1])}")
        print(f"Classes: {self.data['species'].unique()}")
        
        # Separate features and target
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Reshape for CNN (batch_size, channels, features)
        X_train_cnn = X_train_tensor.unsqueeze(1)  # Add channel dimension
        X_test_cnn = X_test_tensor.unsqueeze(1)
        
        self.X_train = X_train_tensor
        self.X_test = X_test_tensor
        self.X_train_cnn = X_train_cnn
        self.X_test_cnn = X_test_cnn
        self.y_train = y_train_tensor
        self.y_test = y_test_tensor
        self.y_encoded = y_encoded
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_cnn_model(self):
        """Create a 1D CNN model for iris classification"""
        print("Creating 1D CNN model...")
        
        model = IrisCNN(input_size=4, num_classes=3).to(self.device)
        
        print("CNN Model Summary:")
        print(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def create_dense_model(self):
        """Create a traditional dense neural network"""
        print("Creating Dense Neural Network...")
        
        model = IrisDense(input_size=4, num_classes=3).to(self.device)
        
        print("Dense Model Summary:")
        print(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_model(self, model, X_train, y_train, model_name="Model", epochs=200):
        """Train the model"""
        print(f"Training {model_name}...")
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        # Training history
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        # Split training data for validation
        val_size = int(0.2 * len(X_train))
        train_size = len(X_train) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            list(zip(X_train, y_train)), [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = correct / total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'best_{model_name.lower()}_model.pth')
            else:
                patience_counter += 1
            
            # Store history
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            if epoch % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] - '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{model_name.lower()}_model.pth'))
        
        history = {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        }
        
        return history
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate the trained model"""
        print(f"\nEvaluating {model_name}...")
        
        model.eval()
        
        with torch.no_grad():
            # Make predictions
            X_test_device = X_test.to(self.device)
            outputs = model(X_test_device)
            _, y_pred = torch.max(outputs, 1)
            
            # Convert to numpy for sklearn metrics
            y_pred_np = y_pred.cpu().numpy()
            y_true_np = y_test.numpy()
            
            # Calculate accuracy
            accuracy = accuracy_score(y_true_np, y_pred_np)
            print(f"Accuracy: {accuracy:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_true_np, y_pred_np, target_names=self.label_encoder.classes_))
            
            # Confusion matrix
            cm = confusion_matrix(y_true_np, y_pred_np)
            print("\nConfusion Matrix:")
            print(cm)
            
            return accuracy, y_pred_np, outputs.cpu().numpy()
    
    def plot_training_history(self, history, model_name="Model"):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history['train_accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history['train_loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def predict_new_sample(self, model, sample):
        """Predict class for a new sample"""
        # Preprocess sample
        sample_scaled = self.scaler.transform([sample])
        sample_tensor = torch.FloatTensor(sample_scaled)
        
        if len(model.conv_layers) > 0:  # CNN model
            sample_tensor = sample_tensor.unsqueeze(1)  # Add channel dimension
        
        model.eval()
        with torch.no_grad():
            sample_device = sample_tensor.to(self.device)
            prediction = model(sample_device)
            probabilities = torch.softmax(prediction, dim=1)
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_prob = torch.max(probabilities).item()
            all_probs = probabilities.cpu().numpy()[0]
        
        predicted_species = self.label_encoder.classes_[predicted_class]
        
        return predicted_species, predicted_prob, all_probs

def main():
    """Main function to run the iris classification"""
    print("=== Iris Flower Classification using PyTorch Neural Networks ===\n")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize classifier
    classifier = IrisClassifier(device=device)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = classifier.load_and_preprocess_data()
    
    print("\n" + "="*50)
    print("1D CNN MODEL")
    print("="*50)
    
    # Create and train CNN model
    cnn_model = classifier.create_cnn_model()
    cnn_history = classifier.train_model(cnn_model, classifier.X_train_cnn, classifier.y_train, "CNN")
    
    # Evaluate CNN model
    cnn_accuracy, cnn_pred, cnn_proba = classifier.evaluate_model(
        cnn_model, classifier.X_test_cnn, classifier.y_test, "CNN"
    )
    
    # Plot CNN results
    classifier.plot_training_history(cnn_history, "CNN")
    classifier.plot_confusion_matrix(
        classifier.y_test.numpy(), cnn_pred, "CNN"
    )
    
    print("\n" + "="*50)
    print("DENSE NEURAL NETWORK")
    print("="*50)
    
    # Create and train dense model
    dense_model = classifier.create_dense_model()
    dense_history = classifier.train_model(dense_model, classifier.X_train, classifier.y_train, "Dense")
    
    # Evaluate dense model
    dense_accuracy, dense_pred, dense_proba = classifier.evaluate_model(
        dense_model, classifier.X_test, classifier.y_test, "Dense"
    )
    
    # Plot dense results
    classifier.plot_training_history(dense_history, "Dense")
    classifier.plot_confusion_matrix(
        classifier.y_test.numpy(), dense_pred, "Dense"
    )
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"CNN Model Accuracy: {cnn_accuracy:.4f}")
    print(f"Dense Model Accuracy: {dense_accuracy:.4f}")
    
    if cnn_accuracy > dense_accuracy:
        print("CNN model performs better!")
        best_model = cnn_model
        best_model_name = "CNN"
    else:
        print("Dense model performs better!")
        best_model = dense_model
        best_model_name = "Dense"
    
    # Test with new samples
    print("\n" + "="*50)
    print("PREDICTION ON NEW SAMPLES")
    print("="*50)
    
    # Sample test cases
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Iris-setosa
        [6.3, 3.3, 4.7, 1.6],  # Iris-versicolor
        [6.3, 3.3, 6.0, 2.5]   # Iris-virginica
    ]
    
    for i, sample in enumerate(test_samples):
        species, prob, all_probs = classifier.predict_new_sample(best_model, sample)
        print(f"\nSample {i+1}: {sample}")
        print(f"Predicted: {species} (Confidence: {prob:.4f})")
        print(f"All probabilities: {dict(zip(classifier.label_encoder.classes_, all_probs))}")
    
    print(f"\nBest model ({best_model_name}) weights saved as 'best_{best_model_name.lower()}_model.pth'")

if __name__ == "__main__":
    main() 