import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml,load_data
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import joblib



logger = get_logger(__name__)

class DataPreprocessor:

    def __init__(self, train_path, test_path, processed_dir, config_path, ohe_path, standard_scaler_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        self.ohe_path = ohe_path
        self.standard_scaler_path =standard_scaler_path

        os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess_data(self, df):
        try:
            
            # Droping unnecessary columns and duplicates
            logger.info("Dropping unnecessary columns.")
            df.drop(columns=['Booking_ID'], inplace=True)
            logger.info("Dropping Duplicates.")
            df.drop_duplicates(inplace=True)
            
            # Drop rows with missing values
            logger.info("Dropping rows with missing values.")
            initial_shape = df.shape
            df.dropna(inplace=True)
            final_shape = df.shape
            logger.info(f"Shape before dropna: {initial_shape}, Shape after dropna: {final_shape}")
            

            return df
    
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise CustomException("Error while preprocessing data.")    


    def balance_data(self, df):
        try:
            logger.info("Balancing the dataset using SMOTE.")
            X = df.drop(columns=[self.config['data_processing']['target_column']])
            y = df[self.config['data_processing']['target_column']]

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_df = pd.concat([X_resampled, y_resampled], axis=1)
            logger.info("Dataset balanced successfully.")
            return balanced_df

        except Exception as e:
            logger.error(f"Error during balancing data: {e}")
            raise CustomException("Error while balancing data.")  
        
        
    def save_processed_data(self, df, file_path):
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Processed data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise CustomException("Error while saving processed data.")
        
    def process(self):
        try:
            logger.info("Starting data processing pipeline.")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            # Listing categorical and numerical columns
            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']

            # Label encode target column
            target_col = self.config['data_processing']['target_column']
            logger.info(f"Label encoding target column: {target_col}")
            train_df[target_col] = train_df[target_col].map({'Canceled': 0, 'Not_Canceled': 1})
            test_df[target_col] = test_df[target_col].map({'Canceled': 0, 'Not_Canceled': 1})



            # Apply One-hot encoding
            logger.info("Applying one-hot encoding to categorical columns.")
            ohe = OneHotEncoder(drop='first', sparse_output=False, dtype=int, handle_unknown='ignore')
            ohe.fit(train_df[cat_cols])
            
            # Transform training data 
            train_ohe_df = ohe.transform(train_df[cat_cols])
            train_ohe_features = pd.DataFrame(train_ohe_df, columns=ohe.get_feature_names_out(cat_cols), index=train_df.index)
            train_df = pd.concat([train_df.drop(columns=cat_cols), train_ohe_features], axis=1)
            


            # Transform test data 
            test_ohe_df = ohe.transform(test_df[cat_cols])
            test_ohe_features = pd.DataFrame(test_ohe_df, columns=ohe.get_feature_names_out(cat_cols), index=test_df.index)
            test_df = pd.concat([test_df.drop(columns=cat_cols), test_ohe_features], axis=1)
            
            # Save the fitted OneHotEncoder
            joblib.dump(ohe, self.ohe_path)
            logger.info(f"OneHotEncoder saved to {self.ohe_path}")

            # Balance training data
            train_df = self.balance_data(train_df)


            # Standard scaling for numerical columns
            num_cols = self.config['data_processing']['numerical_columns']
            num_cols = [col for col in num_cols if col in train_df.columns and col != target_col]
            
            if num_cols:
                logger.info(f"Applying standard scaling to numerical columns: {num_cols}")
                scaler = StandardScaler()
                
                # Fit on training data and transform both
                train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
                
                # Only transform test data if columns exist
                test_num_cols = [col for col in num_cols if col in test_df.columns]
                if test_num_cols:
                    test_df[test_num_cols] = scaler.transform(test_df[test_num_cols])
                
                # Save the fitted StandardScaler
                joblib.dump(scaler, self.standard_scaler_path)
                logger.info(f"StandardScaler saved to {self.standard_scaler_path}")
        
            # Save processed data
            self.save_processed_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_processed_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Error in data processing pipeline: {e}")
            raise CustomException("Error in data processing pipeline.")


if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH,
        ohe_path = OHE_ENCODER_PATH,
        standard_scaler_path = STANDARD_SCALER_PATH
    )
    preprocessor.process()