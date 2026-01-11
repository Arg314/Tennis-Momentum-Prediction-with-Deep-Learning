from src.utils import load_config, set_seed
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import prepare_dataset
from src.train import train_model
from src.evaluate import evaluate_model
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="MultiStepLSTM", help="Model to train")
    args = parser.parse_args()
    
    set_seed(42)
    config = load_config()
    
    # Load data
    df = load_and_clean_data(config)
    
    # Prepare features
    seq_X, scalar_X, y = prepare_dataset(df.head(300), config)  # limit for demo
    
    # Train
    model, val_data = train_model(seq_X, scalar_X, y, config, args.model)
    
    # Evaluate
    metrics = evaluate_model(model, *val_data, config)
    print(f"\n{args.model} Evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()