import pickle
import numpy as np
#I'm going to use SK
from sklearn.neural_network import MLPClassifier 
def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

def train_model():
    training_file = input("Enter the training data file: ")
    X_train, y_train = load_data(training_file)
    
    #Neural network 
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)

    with open('digit_classifier_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Calculate accuracy as a percent
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true) * 100  
    return accuracy

def load_model():
    try:
        with open('digit_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print("No saved model found. Let's train a new one.")
        return train_model()

def main():
    userInput = input("Do you want to train the model? (y/n): ").lower()
    
    if userInput == 'y':
        model = train_model()
    else:
        model = load_model()
    
    test_file = input("Enter the test data file: ")
    X_test, y_test = load_data(test_file)
    
    # Predictions
    predictions = model.predict(X_test)

    # Output true label vs predicted label for each instance
    print("\nTest Data Predictions:")
    for i in range(len(X_test)):
        print(f"Instance {i+1}: Predicted = {int(predictions[i])}")
    
    # Calculate and print accuracy
    accuracy = calculate_accuracy(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()