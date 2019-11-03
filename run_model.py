from utils import load_model
import sys
import numpy as np


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run_model.py model_dir model_prefix")
        sys.exit(1)
    model_dir = sys.argv[1]
    model_prefix = sys.argv[2]
    model, scaler = load_model(model_dir, model_prefix)

    customer_info = scaler.transform( np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000 ]]))
    new_pred = model.predict(customer_info)
    print(new_pred > 0.5)
    
