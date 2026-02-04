
import sys
import numpy as np
import json
import os

def parse_diffusion_trbfile(path):
    if not path.endswith(".trb"):
        raise ValueError(f"Output file must be .trb, got: {path}")

    # Load the pickle
    data_dict = np.load(path, allow_pickle=True)

    # Extract relevant data
    sd = {}
    
    # pLDDT
    if "plddt" in data_dict:
        last_plddts = data_dict["plddt"][-1]
        sd["plddt"] = float(np.mean(last_plddts))
        sd["perres_plddt"] = last_plddts.tolist() # Convert numpy array to list for JSON
    
    # Metadata
    sd["location"] = path.replace(".trb", ".pdb")
    sd["description"] = os.path.basename(path).replace(".trb", "")
    
    if "config" in data_dict and "inference" in data_dict["config"]:
        sd["input_pdb"] = data_dict["config"]["inference"].get("input_pdb")
        
    # Other scores
    scoreterms = ["con_hal_pdb_idx", "con_ref_pdb_idx", "sampled_mask"]
    for st in scoreterms:
        if st in data_dict:
            val = data_dict[st]
            if isinstance(val, np.ndarray):
                sd[st] = val.tolist()
            else:
                sd[st] = val

    return sd

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_trb.py <path_to_trb>")
        sys.exit(1)
        
    trb_path = sys.argv[1]
    if not os.path.exists(trb_path):
        print(json.dumps({"error": f"File not found: {trb_path}"}))
        sys.exit(1)
        
    try:
        data = parse_diffusion_trbfile(trb_path)
        print(json.dumps(data))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
