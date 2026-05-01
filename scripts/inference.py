import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utility_functions import load_model, load_tasks
import grelu.sequence.format

def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    

    chrom = "chr12"
    input_len = 524288
    input_start = 113157856
    input_end = input_start + input_len
    
    input_intervals = pd.DataFrame({
        'chrom':[chrom], 'start':[input_start], 'end':[input_end], "strand":["-"],
    })
    
    # Extract sequence
    print(f"Extracting sequence for {chrom}:{input_start}-{input_end}...")
    input_seqs = grelu.sequence.format.convert_input_type(
        input_intervals,
        output_type="strings",
        genome="mm10"
    )
    
    print("Loading fine-tuned model...")
    ft_model, _ = load_model(
        species="mouse", 
        model_selection="fine_tuned", 
        checkpoint_path="/netfiles/dmlabshare1/rstagg/programmed_delayed_splicing/rstest_outputs/fine_tune/last_fixed.ckpt",
        device=device
    )
    
    print("inference w fine tuned model")
    ft_preds = ft_model.predict_on_seqs(input_seqs, device=device)
    
    out_dir = "/netfiles/dmlabshare1/rstagg/programmed_delayed_splicing/rstest_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    ft_out_path = os.path.join(out_dir, "ft_ighm_preds.npy")
    np.save(ft_out_path, ft_preds)
    print(f"Fine-tuned predictions saved to {ft_out_path}")
    
    # Free up memory
    del ft_model
    torch.cuda.empty_cache()
    
    print("Loading pre-trained model")
    pt_model, _ = load_model(
        species="mouse",
        model_selection="pretrained",
        device=device
    )
    
    print("Running inference w pre-trained model")
    pt_preds = pt_model.predict_on_seqs(input_seqs, device=device)
    
    pt_out_path = os.path.join(out_dir, "pt_ighm_preds.npy")
    np.save(pt_out_path, pt_preds)
    print(f"Pre-trained predictions saved to {pt_out_path}")
    
    print("Inference completed successfully.")

if __name__ == "__main__":
    main()
