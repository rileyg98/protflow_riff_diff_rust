
import sys
import os
import argparse

def main():
    # Rust command: python script options...
    # --out_folder ... --pdb_path ...
    
    out_folder = None
    pdb_path = None
    
    for i, arg in enumerate(sys.argv):
        if arg == "--out_folder":
            out_folder = sys.argv[i+1]
        if arg == "--pdb_path":
            pdb_path = sys.argv[i+1]
            
    if not out_folder or not pdb_path:
        print("Error: missing args")
        sys.exit(1)

    # Create folder (LigandMPNN actually creates it)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    fa_path = os.path.join(out_folder, f"{pdb_name}.fa")
    
    with open(fa_path, "w") as f:
        f.write(f">{pdb_name}, score=0.5, seq_recovery=0.9\n")
        f.write("MKFIV\n")

    print(f"Mock LigandMPNN ran. Output: {fa_path}")

if __name__ == "__main__":
    main()
