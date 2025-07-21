from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *


if __name__ == "__main__":
    
    ### 1. Data Ingestion

    config = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()


    ### 2. Data Processing

    preprocessor = DataPreprocessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH,
        ohe_path = OHE_ENCODER_PATH,
        standard_scaler_path = STANDARD_SCALER_PATH
    )
    preprocessor.process()


    ### 3. Model Training

    trainer = ModelTraining(
        train_path= PROCESSED_TRAIN_DATA_PATH,
        test_path= PROCESSED_TEST_DATA_PATH,
        model_output_path= MODEL_OUTPUT_PATH,
        evaluation_metrics_path= EVALUATION_METRICS_PATH        
    )
    trainer.run()