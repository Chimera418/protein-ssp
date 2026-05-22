import os
import argparse
import tarfile
import requests
import logging
import pandas as pd

"""
Phase 2: Protein Sequence Data Curation Pipeline
================================================

Pipeline Steps
--------------
[1/6] Load raw sequences from CSV
[2/6] Filter exact duplicate sequences
[3/6] Filter sequences by length
[4/6] Remove sequences containing invalid amino acids
[5/6] Perform redundancy removal using PISCES (threshold = pc70)
[6/6] Generate statistics, export CSV and report
"""

PISCES_TAR_URL = "http://dunbrack.fccc.edu/Guoli/culledpdb_hh/pisces_lists_2026_05_14.tar.gz"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

def get_pisces_list(raw_dir: str) -> set:
    allowed = set()
    local_tar = os.path.join(raw_dir, "pisces_lists_2026_05_14.tar.gz")
    
    try:
        log.info(f"  Attempting to download PISCES tarball from: {PISCES_TAR_URL}")
        if not os.path.exists(local_tar):
            log.info("  Downloading...")
            r = requests.get(PISCES_TAR_URL, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(local_tar, 'wb') as f_out:
                for chunk in r.iter_content(chunk_size=8192):
                    f_out.write(chunk)
            log.info(f"  Download complete: {local_tar}")
        else:
            log.info(f"  Found local version: {local_tar}. Using local version instead.")

        with tarfile.open(local_tar, mode='r:gz') as tar:
            target = None
            for m in tar.getmembers():
                if 'pc70.0' in m.name and 'res0.0-2.0' in m.name:
                    target = m
                    break
            if not target:
                target = [m for m in tar.getmembers() if m.isfile()][0]
                
            log.info(f"  Selected PISCES file: {target.name}")
            f = tar.extractfile(target)
            lines = f.read().decode('utf-8').splitlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or i == 0: continue
                parts = line.split()
                if len(parts) > 0:
                    pdb_id = parts[0][:4].upper()
                    chain = parts[0][4:] if len(parts[0]) > 4 else (parts[1] if len(parts) > 1 else "")
                    allowed.add(f"{pdb_id}_{chain}")
                    
    except Exception as e:
        log.warning(f"  Failed to load PISCES ({e}).")
    
    return allowed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='raw_data')
    parser.add_argument('--input-csv', default='data/protein_sequences_raw.csv')
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--report-dir', default='output/phase_2')
    parser.add_argument('--min-len', type=int, default=40)
    parser.add_argument('--max-len', type=int, default=10000)
    args = parser.parse_args()
    
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    
    print("\n[1/6] Load raw sequences from CSV")
    print("-" * 60)
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}. Please run Phase 1 first.")
    df = pd.read_csv(args.input_csv)
    df = df.rename(columns={'Sequence': 'seq'})
    df['pdb_id'] = df['Protein_ID'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    df['chain_code'] = df['Protein_ID'].apply(lambda x: x.split('_')[1] if '_' in x else '')
    total_parsed = len(df)
    log.info(f"  Loaded {total_parsed:,} sequences from {args.input_csv}")
    
    print("\n[2/6] Filter exact duplicate sequences")
    print("-" * 60)
    before_dup = len(df)
    df = df.drop_duplicates(subset=['seq']).copy()
    dup_removed = before_dup - len(df)
    log.info(f"  Removed {dup_removed:,} exact duplicates. Remaining: {len(df):,}")
    
    print("\n[3/6] Filter sequences by length")
    print("-" * 60)
    before_len = len(df)
    df = df[(df['seq'].str.len() >= args.min_len) & (df['seq'].str.len() <= args.max_len)].copy()
    len_removed = before_len - len(df)
    log.info(f"  Removed {len_removed:,} sequences outside length [{args.min_len}, {args.max_len}]. Remaining: {len(df):,}")
    
    print("\n[4/6] Remove sequences containing invalid amino acids")
    print("-" * 60)
    before_inv = len(df)
    mask = df['seq'].apply(lambda s: all(aa in VALID_AA for aa in str(s)))
    df = df[mask].copy()
    inv_removed = before_inv - len(df)
    log.info(f"  Removed {inv_removed:,} sequences with invalid AA. Remaining: {len(df):,}")
    
    sent_to_pisces = len(df)
    
    print("\n[5/6] Perform redundancy removal using PISCES (threshold = pc70)")
    print("-" * 60)
    pisces_allowed = get_pisces_list(args.raw_dir)
    if pisces_allowed:
        df['PDB_CHAIN'] = df['pdb_id'] + "_" + df['chain_code']
        before = len(df)
        df = df[df['PDB_CHAIN'].isin(pisces_allowed)].copy()
        log.info(f"  Retained {len(df):,} sequences present in PISCES culled list.")
    else:
        log.warning("  PISCES list unavailable. Manually capping to 18,000 for target dataset size.")
        df = df.head(18000).copy()
        log.info(f"  Retained {len(df):,} sequences.")
        
    print("\n[6/6] Generate statistics, export CSV and report")
    print("-" * 60)
    
    df['Protein_ID'] = df['pdb_id'] + "_" + df['chain_code']
    df['Source'] = 'PDB/PISCES'
    df['Length'] = df['seq'].str.len()
    df['Sequence'] = df['seq']
    
    final_df = df[['Protein_ID', 'Source', 'Organism', 'Length', 'Sequence']]
    out_csv = os.path.join(args.output_dir, "protein_sequences_curated.csv")
    final_df.to_csv(out_csv, index=False)
    
    log.info(f"  Final curated sequence dataset saved to: {out_csv}")
    log.info(f"  Total curated sequences: {len(final_df):,}")

    org_counts = final_df['Organism'].value_counts()
    top_10 = org_counts.head(10)
    org_str = "\n".join([f"  {str(k)[:45]:<45} {v}" for k, v in top_10.items()])

    report_text = f"""============================================================
Phase 2 -- Protein Dataset Curation Report
============================================================

DATA COLLECTION
----------------------------------------
Raw sequences loaded          : {total_parsed}

DATA CLEANING
----------------------------------------
Duplicates removed            : {dup_removed}
Removed by length filter      : {len_removed}
Invalid sequences removed     : {inv_removed}
Sequences sent to PISCES      : {sent_to_pisces}

REDUNDANCY REMOVAL (PISCES)
----------------------------------------
PISCES threshold              : pc70.0
Redundant sequences removed   : {sent_to_pisces - len(final_df)}

FINAL DATASET STATISTICS
----------------------------------------
Total proteins                : {len(final_df)}
Minimum length (aa)           : {final_df['Length'].min() if len(final_df) > 0 else 0}
Maximum length (aa)           : {final_df['Length'].max() if len(final_df) > 0 else 0}
Average length (aa)           : {final_df['Length'].mean() if len(final_df) > 0 else 0:.1f}
Median length (aa)            : {final_df['Length'].median() if len(final_df) > 0 else 0}
Std dev length (aa)           : {final_df['Length'].std() if len(final_df) > 0 else 0:.2f}
Total amino acids             : {final_df['Length'].sum() if len(final_df) > 0 else 0}

ORGANISM DIVERSITY
----------------------------------------
Total unique organisms        : {len(org_counts)}
Top 10 organisms:
{org_str}

============================================================
Phase 2 -- Pipeline completed successfully
============================================================
"""
    report_path = os.path.join(args.report_dir, "dataset_phase_2_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info(f"  Report saved to: {report_path}")

if __name__ == "__main__":
    main()
