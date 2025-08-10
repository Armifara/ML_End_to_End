# loguru is a logging library that provides an easy-to-use interface for logging in Python applications.
# It allows for flexible logging configurations and is often used in Python projects for better logging management.

from loguru import logger
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from constants import DATA_FILE, MODEL_DIR, MODEL_FILE, TEST_SIZE, RANDOM_STATE, TARGET

def train_and_save_model():
    try:
        # Step 1 - Load the dataset
        logger.info("Loading dataset/Data Ingestion from {}", DATA_FILE)
        df = pd.read_csv(DATA_FILE)
        logger.info("Dataset loaded successfully with shape: {}", df.shape)
    
        # Step 2 - Remove Duplicates
        dup = df.duplicated().sum()
        logger.info("Duplicates found: {}", dup)
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        logger.info("Duplicates removed. New shape: {}", df.shape)

        # Check Missing Values
        m = df.isna().sum()
        logger.info("Missing values found: {}", m.to_dict())

        # Step 3 - Separate X & Y
        logger.info("Separating X and Y")
        X = df.drop(columns=[TARGET])
        Y = df[TARGET]
        logger.info("X shape: {}, Y shape: {}", X.shape, Y.shape)

        # Step 4 - Split the dataset
        logger.info("Splitting dataset into train and test sets with test size: {} and random state = {}", TEST_SIZE, RANDOM_STATE)
        xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        logger.info("XTrain set shape: {}, YTrain set shape: {}", xtrain.shape, ytrain.shape)
        logger.info("XTest set shape: {}, YTest set shape: {}", xtest.shape, ytest.shape)

        # Step 5 - Create a pipeline
        logger.info("Initializing Model Pipeline...")
        model = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LogisticRegression(random_state=RANDOM_STATE)
        )

        # Cross-validation
        logger.info("Performing cross-validation")
        scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring='f1_macro')
        logger.info("Cross-validation scores: {}", scores)
        scores_mean = scores.mean().round(4)
        scores_std = scores.std().round(4)
        logger.info("Cross-validation Mean: {} +/- Standard Deviation: {}", scores_mean, scores_std) 

        # Step 6 - Train the model
        logger.info("Training the model...")
        model.fit(xtrain, ytrain)
        logger.info("Model trained successfully")

        # Step 7 - Evaluate the model
        logger.info("Evaluating the model...")
        ypred_train = model.predict(xtrain)
        ypred_test = model.predict(xtest)
        f1_train = f1_score(ytrain, ypred_train, average='macro')
        f1_test = f1_score(ytest, ypred_test, average='macro')
        logger.info("F1 Score on Train set: {}", f1_train)
        logger.info("F1 Score on Test set: {}", f1_test)

        logger.info("Classification Report on Test set:\n{}", classification_report(ytest, ypred_test))
        
        # Step 8 - Save the model
        # mkdir creates a directory if it does not exist.
        MODEL_DIR.mkdir(exist_ok=True) # Create model directory if it doesn't exist
        joblib.dump(model, MODEL_FILE)
        logger.info("Model saved successfully at {}", MODEL_FILE)
    
        logger.success("Training and model saving completed successfully")

    except Exception as e:
        logger.error("Error loading dataset: {}", e)

if __name__ == "__main__":
    train_and_save_model()
    