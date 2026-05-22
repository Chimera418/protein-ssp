import os
import gzip
import argparse
import datetime
import requests
import logging
import pandas as pd

"""
Phase 1: Raw Protein Sequences Pipeline
=======================================

Pipeline Steps
--------------
[1/3] Collect protein sequences from RCSB (ss.txt.gz)
[2/3] Parse protein sequences
[3/3] Export raw sequence dataset to CSV and generate report
"""

RCSB_SS_URLS = [
    "https://ftp.wwpdb.org/pub/pdb/derived_data/ss.txt.gz",
    "https://files.wwpdb.org/pub/pdb/derived_data/ss.txt.gz",
]
FALLBACK_SS_GZ = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pdb-secondary-structure", "raw_data", "2018-06-06-ss.txt.gz"
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

def download_ss_gz(dest: str):
    target_url = RCSB_SS_URLS[0]
    log.info(f"  Attempting to download RCSB ss.txt.gz from: {target_url}")
    if os.path.exists(dest):
        log.info(f"  Found local version: {dest}. Using local version instead.")
        return
    for url in RCSB_SS_URLS:
        try:
            log.info(f"  Downloading from: {url}")
            r = requests.get(url, stream=True, timeout=10)
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            log.info(f"  Download complete: {dest}")
            return
        except Exception as e:
            log.warning(f"  Failed: {e}")
    log.error("  All RCSB ss.txt.gz sources failed.")
    if os.path.exists(FALLBACK_SS_GZ):
        import shutil
        log.warning(f"  Using bundled fallback: {FALLBACK_SS_GZ}")
        shutil.copy2(FALLBACK_SS_GZ, dest)
    else:
        raise RuntimeError("Could not download ss.txt.gz and no local fallback found.")

def download_source_idx(dest: str):
    url = "https://files.wwpdb.org/pub/pdb/derived_data/index/source.idx"
    log.info(f"  Attempting to download source.idx from: {url}")
    if os.path.exists(dest):
        log.info(f"  Found local version: {dest}. Using local version instead.")
        return
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        log.info(f"  Download complete: {dest}")
    except Exception as e:
        log.warning(f"  Failed to download source.idx: {e}")

def parse_organism_map(idx_path: str) -> dict:
    org_map = {}
    if not os.path.exists(idx_path):
        return org_map
    with open(idx_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                pdb_id, source = parts
                if len(pdb_id) == 4 and pdb_id.isalnum():
                    org_map[pdb_id.upper()] = source.strip().capitalize()
    return org_map

def parse_ss_txt_gz(infile: str) -> pd.DataFrame:
    log.info(f"  Parsing: {infile}")
    records = []
    state = None
    seq_id = seq_chain = seq = ''
    
    with gzip.open(infile, 'rt') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if line.startswith('>'):
                header = line.lstrip('>').strip()
                parts = header.split(':')
                pid = parts[0].strip().upper()
                chain = parts[1].strip() if len(parts) > 1 else ''
                
                if state == 'ss' or state == 'seq':
                    if state == 'seq' or state == 'ss':
                        if len(seq) > 0 and seq_id != '':
                            records.append({
                                'pdb_id': seq_id,
                                'chain_code': seq_chain,
                                'seq': seq
                            })
                
                if 'sequence' in line.lower() or len(parts) == 4:
                    state = 'seq'
                    seq_id = pid
                    seq_chain = chain
                    seq = ''
                else:
                    state = 'ss'
                continue
            
            if state == 'seq':
                seq += line.strip()
                
    if state == 'seq' and len(seq) > 0 and seq_id != '':
         records.append({'pdb_id': seq_id, 'chain_code': seq_chain, 'seq': seq})
         
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=['pdb_id', 'chain_code'])
    log.info(f"  Parsed {len(df):,} sequences.")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='raw_data')
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--report-dir', default='output/phase_1')
    args = parser.parse_args()
    
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    date_str = datetime.date.today().strftime('%Y-%m-%d')
    ss_gz_path = os.path.join(args.raw_dir, f"{date_str}-ss.txt.gz")
    source_idx_path = os.path.join(args.raw_dir, f"{date_str}-source.idx")
    
    print("\n[1/3] Collect protein sequences and metadata from RCSB")
    print("-" * 60)
    download_ss_gz(ss_gz_path)
    download_source_idx(source_idx_path)
    org_map = parse_organism_map(source_idx_path)
    log.info(f"  Parsed organism data for {len(org_map):,} PDBs.")
    
    print("\n[2/3] Parse protein sequences")
    print("-" * 60)
    df = parse_ss_txt_gz(ss_gz_path)
    total_parsed = len(df)
    
    print("\n[3/3] Export raw sequence dataset to CSV and generate report")
    print("-" * 60)
    
    df['Protein_ID'] = df['pdb_id'] + "_" + df['chain_code']
    df['Source'] = 'PDB/RCSB'
    df['Organism'] = df['pdb_id'].map(org_map).fillna('Unknown')
    df['Length'] = df['seq'].str.len()
    df['Sequence'] = df['seq']
    
    final_df = df[['Protein_ID', 'Source', 'Organism', 'Length', 'Sequence']]
    out_csv = os.path.join(args.output_dir, "protein_sequences_raw.csv")
    final_df.to_csv(out_csv, index=False)
    
    log.info(f"  Raw sequence dataset saved to: {out_csv}")
    log.info(f"  Total raw sequences: {len(final_df):,}")

    org_counts = final_df['Organism'].value_counts()
    top_10 = org_counts.head(10)
    org_str = "\n".join([f"  {str(k)[:45]:<45} {v}" for k, v in top_10.items()])

    report_text = f"""============================================================
Phase 1 -- Raw Protein Sequences Report
============================================================

DATA COLLECTION
----------------------------------------
RCSB PDB sequences parsed     : {total_parsed}
Total sequences collected     : {total_parsed}

DATASET STATISTICS
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
Phase 1 -- Pipeline completed successfully
============================================================
"""
    report_path = os.path.join(args.report_dir, "dataset_phase_1_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info(f"  Report saved to: {report_path}")

if __name__ == "__main__":
    main()
