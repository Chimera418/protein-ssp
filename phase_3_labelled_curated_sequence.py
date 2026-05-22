import os
import csv
import gzip
import argparse
import datetime
import logging
import pandas as pd

"""
Phase 3: Protein Sequence Labelling Pipeline
============================================

Pipeline Steps
--------------
[1/5] Load Phase 2 curated sequences (protein_sequences_curated.csv)
[2/5] Parse secondary structure labels from RCSB (ss.txt.gz)
[3/5] Map labels to curated sequences
[4/5] Convert SST8 to SST3 labels
[5/5] Export phase 3 labelled dataset to CSV
"""

SST8_TO_SST3 = {
    'H': 'H', 'G': 'H', 'I': 'H',
    'E': 'E', 'B': 'E',
    'T': 'C', 'S': 'C', 'C': 'C'
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

def parse_labels(infile: str, target_ids: set) -> dict:
    log.info(f"  Parsing labels from: {infile}")
    labels = {}
    state = None
    seq_id = seq_chain = ''
    sst = ''
    
    with gzip.open(infile, 'rt') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if line.startswith('>'):
                header = line.lstrip('>').strip()
                parts = header.split(':')
                pid = parts[0].strip().upper()
                chain = parts[1].strip() if len(parts) > 1 else ''
                
                if state == 'ss':
                    key = f"{seq_id}_{seq_chain}"
                    if key in target_ids:
                        sst_clean = sst.replace(' ', 'C')
                        labels[key] = sst_clean
                        
                if 'secstr' in line.lower() or (len(parts) == 4 and 'secstr' in parts[2]):
                    state = 'ss'
                    seq_id = pid
                    seq_chain = chain
                    sst = ''
                else:
                    state = 'seq'
                continue
                
            if state == 'ss':
                sst += line
                
    if state == 'ss':
        key = f"{seq_id}_{seq_chain}"
        if key in target_ids:
            sst_clean = sst.replace(' ', 'C')
            labels[key] = sst_clean
            
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='raw_data')
    parser.add_argument('--input-csv', default='data/protein_sequences_curated.csv')
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--report-dir', default='output/phase_3')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    date_str = datetime.date.today().strftime('%Y-%m-%d')
    ss_gz_path = os.path.join(args.raw_dir, f"{date_str}-ss.txt.gz")
    
    print("\n[1/5] Load Phase 2 curated sequences")
    print("-" * 60)
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}. Please run Phase 2 first.")
    df = pd.read_csv(args.input_csv)
    log.info(f"  Loaded {len(df):,} sequences from {args.input_csv}")
    
    target_ids = set(df['Protein_ID'])
    
    print("\n[2/5] Parse secondary structure labels from RCSB (ss.txt.gz)")
    print("-" * 60)
    if not os.path.exists(ss_gz_path):
        fallback = os.path.join("pdb-secondary-structure", "raw_data", "2018-06-06-ss.txt.gz")
        if os.path.exists(fallback):
            log.warning("  Using bundled fallback ss.txt.gz")
            ss_gz_path = fallback
        else:
            raise FileNotFoundError("ss.txt.gz not found! Run Phase 1 to download it.")
            
    labels_dict = parse_labels(ss_gz_path, target_ids)
    log.info(f"  Found labels for {len(labels_dict):,} target sequences.")
    
    print("\n[3/5] Map labels to curated sequences")
    print("-" * 60)
    df['sst8'] = df['Protein_ID'].map(labels_dict)
    
    before = len(df)
    df = df.dropna(subset=['sst8']).copy()
    df = df[df['Sequence'].str.len() == df['sst8'].str.len()].copy()
    removed_mismatch = before - len(df)
    log.info(f"  Removed {removed_mismatch:,} sequences due to missing labels or length mismatch. Remaining: {len(df):,}")
    
    print("\n[4/5] Convert SST8 to SST3 labels")
    print("-" * 60)
    def to_sst3(sst8):
        return ''.join(SST8_TO_SST3.get(c, 'C') for c in sst8)
    df['sst3'] = df['sst8'].apply(to_sst3)
    
    print("\n[5/5] Export phase 3 labelled dataset to CSV")
    print("-" * 60)
    df['pdb_id'] = df['Protein_ID'].apply(lambda x: x.split('_')[0])
    df['chain_code'] = df['Protein_ID'].apply(lambda x: x.split('_')[1] if '_' in x else '')
    df['seq'] = df['Sequence']
    
    final_df = df[['pdb_id', 'chain_code', 'seq', 'sst8', 'sst3']]
    out_csv = os.path.join(args.output_dir, "protein_labelled_curated.csv")
    final_df.to_csv(out_csv, index=False)
    
    log.info(f"  Final labelled dataset saved to: {out_csv}")
    log.info(f"  Total sequences: {len(final_df):,}")

    report_text = f"""============================================================
Phase 3 -- Protein Sequence Labelling Report
============================================================

DATA MAPPING
----------------------------------------
Curated sequences loaded      : {before}
Labels successfully mapped    : {len(final_df)}
Removed (mismatch/missing)    : {removed_mismatch}

FINAL DATASET STATISTICS
----------------------------------------
Total proteins                : {len(final_df)}
Minimum length (aa)           : {final_df['seq'].str.len().min() if len(final_df) > 0 else 0}
Maximum length (aa)           : {final_df['seq'].str.len().max() if len(final_df) > 0 else 0}
Average length (aa)           : {final_df['seq'].str.len().mean() if len(final_df) > 0 else 0:.1f}
Median length (aa)            : {final_df['seq'].str.len().median() if len(final_df) > 0 else 0}

============================================================
Phase 3 -- Pipeline completed successfully
============================================================
"""
    report_path = os.path.join(args.report_dir, "dataset_phase_3_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info(f"  Report saved to: {report_path}")

if __name__ == "__main__":
    main()
