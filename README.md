# Iris Flower Classification using Neural Networks

This project implements iris flower classification using both 1D Convolutional Neural Networks (CNN) and traditional Dense Neural Networks. The goal is to classify iris flowers into three species based on four numerical features: sepal length, sepal width, petal length, and petal width.

## Dataset

The project uses the classic IRIS dataset containing:
- **150 samples** (50 per class)
- **4 features**: sepal_length, sepal_width, petal_length, petal_width
- **3 classes**: Iris-setosa, Iris-versicolor, Iris-virginica

## Features

### 1D CNN Model
- Treats the 4 features as a 1D signal
- Uses Conv1D layers with batch normalization and dropout
- Global average pooling for feature extraction
- Dense layers for final classification

### Dense Neural Network
- Traditional feedforward neural network
- Multiple hidden layers with batch normalization
- Dropout regularization to prevent overfitting
- Optimized for tabular data

### Model Training Features
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Stratified train-test split (80-20)
- Feature scaling using StandardScaler
- Comprehensive evaluation metrics

## Installation

1. Clone or download the project files
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main classification script:
```bash
python iris_cnn_classification.py
```

The script will:
1. Load and preprocess the IRIS.csv dataset
2. Train both CNN and Dense models
3. Evaluate and compare model performance
4. Generate training history plots
5. Display confusion matrices
6. Test predictions on sample data
7. Save the best performing model

## Output

The script provides:
- **Training progress** for both models
- **Model summaries** with layer information
- **Performance metrics**: accuracy, classification report, confusion matrix
- **Training history plots**: accuracy and loss over epochs
- **Confusion matrix visualizations**
- **Model comparison** showing which performs better
- **Sample predictions** on new data
- **Saved model file** (.h5 format)

## Model Architecture

### CNN Model
```
Input(4,1) → Conv1D(32,2) → Conv1D(64,2) → GlobalAvgPool → Dense(128) → Dense(64) → Output(3)
```

### Dense Model
```
Input(4) → Dense(64) → Dense(128) → Dense(64) → Dense(32) → Output(3)
```

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- pandas, numpy, matplotlib, seaborn
- scikit-learn

## File Structure

```
iris_flower_classification/
├── IRIS.csv                    # Dataset file
├── iris_cnn_classification.py  # Main classification script
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Expected Results

Both models typically achieve:
- **Accuracy**: 95-100%
- **Fast convergence** due to the well-separated nature of iris classes
- **Good generalization** on test data

The dense neural network often performs slightly better for this tabular dataset, while the CNN demonstrates how convolutional layers can be adapted for 1D feature sequences.

## Customization

You can modify:
- Model architectures in `create_cnn_model()` and `create_dense_model()`
- Training parameters (epochs, batch size, learning rate)
- Data preprocessing steps
- Evaluation metrics

##Contact :

email : jbarathvishnu2005@gmail.com
