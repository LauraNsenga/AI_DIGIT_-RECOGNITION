import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate
import seaborn as sns
import pandas as pd

def load_data(filename):
    """Load and normalize digit data from CSV"""
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    return X, y

def visualize_samples(X, y, n_samples=10):
    """Display sample digits from dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Sample Handwritten Digits')
    
    for i, ax in enumerate(axes.flat):
        # Reshape flat array to 28x28 image
        img = X[i].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {int(y[i])}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_digits.png')
    print("✓ Saved sample visualizations to 'sample_digits.png'")

def train_model_with_validation(X_train, y_train):
    """Train model with cross-validation"""
    print("\n🔄 Training model with cross-validation...")
    
    # Try multiple architectures
    architectures = [
        (50,),
        (100,),
        (100, 50),
        (128, 64, 32)
    ]
    
    best_model = None
    best_score = 0
    best_arch = None
    
    for arch in architectures:
        model = MLPClassifier(
            hidden_layer_sizes=arch,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Cross-validation
        cv_results = cross_validate(
            model, X_train, y_train, 
            cv=5, 
            scoring='accuracy',
            return_train_score=True
        )
        
        mean_score = cv_results['test_score'].mean()
        print(f"Architecture {arch}: {mean_score:.4f} (+/- {cv_results['test_score'].std():.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_arch = arch
            best_model = model
    
    print(f"\n✓ Best architecture: {best_arch} with CV accuracy: {best_score:.4f}")
    
    # Train final model on all training data
    best_model.fit(X_train, y_train)
    
    return best_model, best_arch, best_score

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    print("\n📊 Evaluating model performance...")
    
    predictions = model.predict(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("✓ Saved confusion matrix to 'confusion_matrix.png'")
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for digit, acc in enumerate(per_class_acc):
        print(f"Digit {digit}: {acc:.2%}")
    
    # Overall metrics
    accuracy = np.sum(predictions == y_test) / len(y_test) * 100
    
    # Find misclassified examples
    misclassified = np.where(predictions != y_test)[0]
    print(f"\n✓ Total misclassified: {len(misclassified)} out of {len(y_test)}")
    
    return accuracy, predictions, misclassified

def visualize_errors(X_test, y_test, predictions, misclassified, n_errors=10):
    """Visualize misclassified examples"""
    if len(misclassified) == 0:
        print("No errors to visualize!")
        return
    
    n_show = min(n_errors, len(misclassified))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Misclassified Examples')
    
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            break
        
        idx = misclassified[i]
        img = X_test[idx].reshape(28, 28)
        true_label = int(y_test[idx])
        pred_label = int(predictions[idx])
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', color='red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    print("✓ Saved misclassification analysis to 'misclassified_examples.png'")

def main():
    print("=" * 60)
    print("HANDWRITTEN DIGIT RECOGNITION SYSTEM")
    print("=" * 60)
    
    # Load training data
    training_file = input("\nEnter the training data file (e.g., mnist_train.csv): ").strip()
    X_train, y_train = load_data(training_file)
    print(f"✓ Loaded {len(X_train)} training samples")
    
    # Visualize samples
    visualize_samples(X_train, y_train)
    
    # Train or load model
    user_input = input("\nDo you want to train a new model? (y/n): ").lower()
    
    if user_input == 'y':
        model, best_arch, cv_score = train_model_with_validation(X_train, y_train)
        
        # Save model with metadata
        model_data = {
            'model': model,
            'architecture': best_arch,
            'cv_accuracy': cv_score,
            'training_samples': len(X_train)
        }
        
        with open('digit_classifier_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n✓ Model saved to 'digit_classifier_model.pkl'")
    else:
        try:
            with open('digit_classifier_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            model = model_data['model']
            print(f"✓ Loaded saved model (Architecture: {model_data['architecture']})")
        except FileNotFoundError:
            print("❌ No saved model found. Training new model...")
            model, best_arch, cv_score = train_model_with_validation(X_train, y_train)
    
    # Test model
    test_file = input("\nEnter the test data file (e.g., mnist_test.csv): ").strip()
    X_test, y_test = load_data(test_file)
    print(f"✓ Loaded {len(X_test)} test samples")
    
    # Evaluate
    accuracy, predictions, misclassified = evaluate_model(model, X_test, y_test)
    
    # Visualize errors
    visualize_errors(X_test, y_test, predictions, misclassified)
    
    print(f"\n{'=' * 60}")
    print(f"FINAL ACCURACY: {accuracy:.2f}%")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
