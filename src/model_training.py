import os
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from config.model_param import *
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path, evaluation_metrics_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.evaluation_metrics_path = evaluation_metrics_path

        self.param_dist = MODEL_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info("Loading training and testing data.")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            logger.info("Data loaded and split into features and target.")

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error loading and splitting data: {e}")
            raise CustomException("Failed to load and split data")
        
    def train_model(self, X_train, y_train):
        try:
            logger.info("Starting model training.")

            xgb_model = XGBClassifier(random_state = self.random_search_params['random_state'])

            logger.info("Starting hyperparameter tuning.")

            random_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=self.param_dist,
                n_iter=self.random_search_params['n_iter'],
                scoring=self.random_search_params['scoring'],
                cv=StratifiedKFold(n_splits=self.random_search_params['cv'], shuffle=True, random_state=self.random_search_params['random_state']),
                verbose=self.random_search_params['verbose'],
                n_jobs=self.random_search_params['n_jobs'],
                random_state=self.random_search_params['random_state']
            )

            random_search.fit(X_train, y_train)

            logger.info("Hyperparameter tuning completed.")

            best_params = random_search.best_params_
            best_model = random_search.best_estimator_
            
            logger.info(f"Best parameters found: {best_params}")

            return best_model
        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException("Failed to train model")
        

    def evaluate_model(self, model, X, y, dataset_name="test"):
        try:
            logger.info(f"Evaluating the model on {dataset_name} data.")

            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]

            accuracy = accuracy_score(y, y_pred)
            recall = recall_score(y, y_pred)
            precision = precision_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            roc_auc = roc_auc_score(y, y_pred_proba)

            logger.info(f"Model evaluation results for {dataset_name} - Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}, ROC AUC: {roc_auc}")

            return {
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
        
        except Exception as e:
            logger.error(f"Error during model evaluation on {dataset_name} data: {e}")
            raise CustomException(f"Failed to evaluate model on {dataset_name} data")
        
    def save_evaluation_results(self, train_metrics, test_metrics):
        try:
            logger.info("Saving evaluation metrics for both train and test sets.")
            os.makedirs(os.path.dirname(self.evaluation_metrics_path), exist_ok=True)
            
            train_metrics_df = pd.DataFrame([train_metrics])
            train_metrics_df['dataset'] = 'train'
            
            test_metrics_df = pd.DataFrame([test_metrics])
            test_metrics_df['dataset'] = 'test'

            combined_metrics_df = pd.concat([train_metrics_df, test_metrics_df], ignore_index=True)
            
            cols = ['dataset'] + [col for col in combined_metrics_df.columns if col != 'dataset']
            combined_metrics_df = combined_metrics_df[cols]

            combined_metrics_df.to_csv(self.evaluation_metrics_path, index=False)

            logger.info(f"Evaluation metrics saved to {self.evaluation_metrics_path}")

        except Exception as e:
            logger.error(f"Error saving evaluation metrics: {e}")
            raise CustomException("Failed to save evaluation metrics")

    def save_model(self, model):
        try:
            logger.info("Saving the trained model.")
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            joblib.dump(model, self.model_output_path)

            logger.info(f"Model saved to {self.model_output_path}")

        except Exception as e:
            logger.error(f"Error saving the model: {e}")
            raise CustomException("Failed to save model")   
        

    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training pipeline.")
                logger.info("Starting MLFLOW Experimentation.")

                logger.info("Logging training and testing dataset to MLFLOW.")
                mlflow.log_artifact(self.train_path, artifact_path="train_data")
                mlflow.log_artifact(self.test_path, artifact_path="test_data")

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                model = self.train_model(X_train, y_train)
                
                train_evaluation_results = self.evaluate_model(model, X_train, y_train, dataset_name="train")
                test_evaluation_results = self.evaluate_model(model, X_test, y_test, dataset_name="test")
                
                self.save_model(model)
                self.save_evaluation_results(train_evaluation_results, test_evaluation_results)

                logger.info("Logging the model into MLFLOW.")
                mlflow.log_artifact(self.model_output_path, artifact_path="model")   

                logger.info("Logging model parameters and evaluation metrics to MLFLOW.")
                mlflow.log_params(model.get_params())
                
                train_metrics_for_mlflow = {f"train_{k}": v for k, v in train_evaluation_results.items()}
                test_metrics_for_mlflow = {f"test_{k}": v for k, v in test_evaluation_results.items()}
                
                mlflow.log_metrics(train_metrics_for_mlflow)
                mlflow.log_metrics(test_metrics_for_mlflow)             

                logger.info("Model training pipeline completed successfully.")

        
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException("Error in model training pipeline")
        
        finally:
            logger.info("Model training pipeline finished.")

if __name__ == "__main__":
    trainer = ModelTraining(
        train_path= PROCESSED_TRAIN_DATA_PATH,
        test_path= PROCESSED_TEST_DATA_PATH,
        model_output_path= MODEL_OUTPUT_PATH,
        evaluation_metrics_path= EVALUATION_METRICS_PATH
    )
    trainer.run()