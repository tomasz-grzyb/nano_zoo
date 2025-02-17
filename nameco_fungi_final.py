#!/usr/bin/env python3

# Standard library imports
import os
import re
import sys
import glob
import gzip
import time
import random
import logging
import argparse
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import Counter
from itertools import product
import umap
import hdbscan
from tqdm import tqdm

def setup_logging(log_dir):
    """Set up logging with file and console output"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'fungal_pipeline.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_command(cmd, logger=None):
    """Execute shell command with robust error handling"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if logger:
            if result.stderr:
                logger.warning(f"Command stderr: {result.stderr.strip()}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"Command failed: {cmd}")
            logger.error(f"Error output: {e.stderr}")
        raise

def log_checker(log, files_to_check):
    """Check if all required files exist and are marked as done in log"""
    if not os.path.exists(log):
        return False
    
    # Check log file for completion markers
    with open(log, 'rt') as f:
        completed = [l.strip() for l in f if 'done. Enjoy' in l]
    
    # Check if all required files exist
    files_exist = all(os.path.exists(f) and os.path.getsize(f) > 0 for f in files_to_check)
    
    return len(completed) > 0 and files_exist

def process_kmer_chunk(chunk_data):
    """Process a chunk of sequences for k-mer counting"""
    chunk_records, kmers, chunk_id = chunk_data
    chunk_counts = []
    for record in chunk_records:
        seq = str(record.seq)
        kmer_counts = [len(re.findall(f'(?={mer})', seq)) for mer in kmers]
        chunk_counts.append([record.id] + kmer_counts)
    return chunk_counts

def kmer_counter(output_dir, input_dir, samples, threads, kmer_length=6, logger=None):
    """Enhanced k-mer counting with progress tracking"""
    if logger:
        logger.info(f"Starting {kmer_length}-mer counting with {threads} threads")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if all samples are already processed
    all_processed = True
    for sample in samples:
        kmer_file = os.path.join(output_dir, sample, 'kmers.tsv')
        if not os.path.exists(kmer_file) or os.path.getsize(kmer_file) == 0:
            all_processed = False
            break
    
    if all_processed:
        if logger:
            logger.info("All k-mer files already exist, skipping k-mer counting")
        return
    
    # Pre-generate k-mers
    nucleotides = 'ACGT'
    kmers = [''.join(c) for c in product(nucleotides, repeat=kmer_length)]
    if logger:
        logger.info(f"Generated {len(kmers)} possible {kmer_length}-mers")
        logger.info(f"Processing {len(samples)} samples in total")
    
    for sample_idx, sample in enumerate(samples, 1):
        sample_output_dir = os.path.join(output_dir, sample)
        kmer_file = os.path.join(sample_output_dir, 'kmers.tsv')
        
        # Check if k-mer file already exists
        if os.path.exists(kmer_file) and os.path.getsize(kmer_file) > 0:
            if logger:
                logger.info(f"\nSample {sample_idx}/{len(samples)}: {sample} - k-mers already exist, skipping")
            continue
            
        sample_start = time.time()
        if logger:
            logger.info(f"\nProcessing sample {sample_idx}/{len(samples)}: {sample}")
        
        input_files = glob.glob(os.path.join(input_dir, f"{sample}*.f*q*"))
        if not input_files:
            if logger:
                logger.warning(f"No input file found for sample {sample}")
            continue
        
        input_file = input_files[0]
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # Read sequences
        open_func = gzip.open if input_file.endswith('.gz') else open
        with open_func(input_file, 'rt') as f:
            records = list(SeqIO.parse(f, 'fastq'))
        
        total_reads = len(records)
        if logger:
            logger.info(f"Found {total_reads:,} reads in {sample}")
        
        # Process in chunks
        chunk_size = 1000
        chunks = []
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            chunks.append((chunk, kmers, i//chunk_size))
        
        # Parallel processing
        results = []
        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(process_kmer_chunk, chunk_data) 
                      for chunk_data in chunks]
            
            for future in tqdm(as_completed(futures), 
                             total=len(futures),
                             desc=f"Processing {sample}",
                             unit="chunk"):
                results.extend(future.result())
        
        # Save results
        kmer_df = pd.DataFrame(results, columns=['ID'] + kmers)
        kmer_df.to_csv(os.path.join(sample_output_dir, 'kmers.tsv'), 
                      sep='\t', index=False)
        
        elapsed = time.time() - sample_start
        if logger:
            logger.info(f"Completed {sample} in {elapsed:.2f} seconds")
            logger.info(f"Processing speed: {total_reads/elapsed:.2f} reads/second")

            
def plot_clusters(labels, embedding, sample, output_file):
    """Create cluster visualization plot"""
    plt.figure(figsize=(10, 10))
    
    # Plot noise points
    noise_mask = labels == -1
    plt.scatter(embedding[noise_mask, 0], 
               embedding[noise_mask, 1],
               color='lightgray',
               s=10,
               alpha=0.5,
               label='Noise')
    
    # Plot clustered points
    clustered_mask = ~noise_mask
    plt.scatter(embedding[clustered_mask, 0],
               embedding[clustered_mask, 1],
               c=labels[clustered_mask],
               cmap='Spectral',
               s=10)
    
    plt.title(f'UMAP + HDBSCAN Clustering: {sample}')
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def cluster_sequences(kmer_dir, output_dir, samples, threads, min_cluster_size=25, 
                     select_epsilon=0.3, logger=None):
    """UMAP + HDBSCAN clustering optimized for fungal ITS"""
    if logger:
        logger.info("Starting UMAP + HDBSCAN clustering")
        logger.info(f"Processing {len(samples)} samples with min_cluster_size={min_cluster_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if all samples are already clustered
    all_clustered = True
    for sample in samples:
        cluster_file = os.path.join(output_dir, sample, 'clusters.tsv')
        if not os.path.exists(cluster_file) or os.path.getsize(cluster_file) == 0:
            all_clustered = False
            break
    
    if all_clustered:
        if logger:
            logger.info("All samples already clustered, skipping clustering")
        return
    
    for sample_idx, sample in enumerate(samples, 1):
        sample_dir = os.path.join(output_dir, sample)
        os.makedirs(sample_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(sample_dir, 'clusters.tsv')):
            if logger:
                logger.info(f"\nSample {sample} clusters already exist, skipping")
            continue
            
        if logger:
            logger.info(f"\nProcessing clustering for {sample} ({sample_idx}/{len(samples)})")
        
        # Read k-mer data
        kmer_file = os.path.join(kmer_dir, sample, 'kmers.tsv')
        data = pd.read_csv(kmer_file, sep='\t', index_col=0)
        
        # UMAP reduction with original ITS parameters
        if logger:
            logger.info("Running UMAP dimensionality reduction")
        
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.075,
            metric='cosine',
            n_components=2,
            random_state=42,
            n_jobs=threads,
            local_connectivity=1.2,
            repulsion_strength=1.0
        )
        
        embedding = reducer.fit_transform(data.values)
        
        # HDBSCAN clustering
        if logger:
            logger.info("Running HDBSCAN clustering")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=select_epsilon,
            cluster_selection_method='eom',
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(embedding)
        
        clusters = pd.DataFrame({'Feature': data.index, 'Cluster': labels})
        clusters = clusters.loc[clusters.Cluster >= 0]
        clusters.Cluster = 'Cluster_' + clusters.Cluster.astype(str)
        
        # Pierwszy subsampling - do porównań między klastrami
        for cid in clusters.Cluster.unique():
            sub = clusters.loc[clusters.Cluster == cid].copy()
            if len(sub) > 60:  # Zmniejszone dla ITS
                sub = sub.sample(n=60, random_state=42)
            data.loc[sub.Feature.tolist(),'FullID'] = sample+'___'+cid+'___'
            
        data = data[data['FullID'].notna()]
        data.FullID = data.FullID + data.index.astype(str)
        data.set_index('FullID', inplace=True)
        data.to_csv(os.path.join(sample_dir, 'subsampled_ids.tsv'), sep='\t')
        clusters.to_csv(os.path.join(sample_dir, 'clusters.tsv'), sep='\t', index=False)
        
        # Plot results
        plot_clusters(labels, embedding, sample, 
                     os.path.join(sample_dir, 'clusters.png'))
        
        if logger:
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"Found {n_clusters} clusters in {sample}")

def shared_clusters(cluster_dir, output_dir, samples, subsample_size=150, random_state=42, logger=None):
    """Pool shared clusters between samples with proper size tracking"""
    if logger:
        logger.info('\nPooling shared clusters...')
    
    os.makedirs(output_dir, exist_ok=True)
    subsampled_dir = os.path.join(output_dir, 'Clusters_subsampled')
    os.makedirs(subsampled_dir, exist_ok=True)
    
    # Initialize dataframe for true cluster counts across samples
    all_clusters = []
    cluster_index = 0
    cluster_mapping = {}  # To map original cluster names to new standardized names
    
    # First pass - collect all clusters and their true sizes
    for sample in samples:
        if logger:
            logger.info(f"Processing clusters from sample: {sample}")
            
        cluster_file = os.path.join(cluster_dir, sample, 'clusters.tsv')
        clusters_df = pd.read_csv(cluster_file, sep='\t')
        
        for original_cluster_id in clusters_df['Cluster'].unique():
            if original_cluster_id == -1:  # Skip noise points
                continue
                
            # Get all reads in this cluster (before subsampling)
            cluster_reads = clusters_df[clusters_df['Cluster'] == original_cluster_id]['ID'].tolist()
            if len(cluster_reads) < 3:  # Skip very small clusters
                continue
            
            # Create standardized cluster name
            new_cluster_name = f'Cluster_{cluster_index}'
            cluster_mapping[f"{sample}_{original_cluster_id}"] = new_cluster_name
            cluster_index += 1
            
            # Store original size
            all_clusters.append({
                'Cluster': new_cluster_name,
                'Sample': sample,
                'Size': len(cluster_reads),  # Store TRUE size
                'Reads': cluster_reads  # Keep reads for subsampling later
            })
    
    # Create counts matrix
    unique_clusters = sorted(set(entry['Cluster'] for entry in all_clusters))
    counts_df = pd.DataFrame(0, index=unique_clusters, columns=samples)
    
    # Fill in the counts matrix with TRUE sizes
    for entry in all_clusters:
        counts_df.loc[entry['Cluster'], entry['Sample']] = entry['Size']
    
    # Now handle subsampling for further processing
    for entry in all_clusters:
        cluster_reads = entry['Reads']
        cluster_name = entry['Cluster']
        
        # Subsample if needed (only for further processing)
        if len(cluster_reads) > subsample_size:
            random.seed(random_state)
            subsampled_reads = random.sample(cluster_reads, subsample_size)
        else:
            subsampled_reads = cluster_reads
        
        # Save subsampled reads for further processing
        output_file = os.path.join(subsampled_dir, f'{cluster_name}.txt')
        with gzip.open(output_file + '.gz', 'wt') as f:
            f.write('\n'.join(subsampled_reads))
    
    # Save the true counts
    counts_df.index.name = 'Cluster'
    counts_df.to_csv(os.path.join(output_dir, 'cluster_counts.tsv'), sep='\t')
    
    # Save mapping information for reference
    mapping_df = pd.DataFrame(list(cluster_mapping.items()), columns=['Original', 'Standardized'])
    mapping_df.to_csv(os.path.join(output_dir, 'cluster_mapping.tsv'), sep='\t', index=False)
    
    if logger:
        logger.info(f"Total clusters processed: {len(unique_clusters)}")
        logger.info(f"Cluster counts saved to: {os.path.join(output_dir, 'cluster_counts.tsv')}")
        logger.info(f"Subsampled cluster files saved to: {subsampled_dir}")
        
    return counts_df

def fq_by_cluster(input_dir, shared_dir, consensus_dir, threads, logger=None):
    """Split fastq by clusters and generate initial consensus"""
    if logger:
        logger.info('\nCreating fastq files for clusters and generating consensus...')
    
    os.makedirs(consensus_dir, exist_ok=True)
    
    # Check if consensus already generated
    consensus_pooled = os.path.join(consensus_dir, 'consensus_pooled.fa')
    if os.path.exists(consensus_pooled) and os.path.getsize(consensus_pooled) > 0:
        if logger:
            logger.info("Consensus sequences already generated, skipping")
        return
    
     
    # Create pooled fastq if doesn't exist
    pooled_fastq = os.path.join(shared_dir, 'pooled.fq')
    if not os.path.exists(pooled_fastq):
        if logger:
            logger.info("Creating pooled FASTQ file...")
        input_files = glob.glob(os.path.join(input_dir, "*.f*q*"))
        with open(pooled_fastq, 'w') as outfile:
            for fq in input_files:
                if fq.endswith('.gz'):
                    with gzip.open(fq, 'rt') as infile:
                        outfile.write(infile.read())
                else:
                    with open(fq, 'rt') as infile:
                        outfile.write(infile.read())
    
    # Process each cluster
    cluster_files = glob.glob(os.path.join(shared_dir, 'Clusters_subsampled', '*.txt'))
    
    for cluster_file in tqdm(cluster_files, desc="Processing clusters"):
        cluster_name = os.path.basename(cluster_file).replace('.txt', '')
        
        # Extract reads for this cluster
        cluster_fq = os.path.join(consensus_dir, f'{cluster_name}.fq')
        cmd = f"grep -A 3 -F -f {cluster_file} {pooled_fastq} | grep -v '^--$' > {cluster_fq}"
        run_command(cmd, logger)
        
        # Generate consensus using SPOA
        consensus_file = os.path.join(consensus_dir, f'{cluster_name}_consensus.fa')
        cmd = f"spoa {cluster_fq} > {consensus_file}"
        run_command(cmd, logger)
        
        # Compress cluster fastq instead of removing it
        cmd = f"gzip -f {cluster_fq}"
        run_command(cmd, logger)
    
    # Combine all consensuses
    with open(consensus_pooled, 'w') as outfile:
        for consensus in glob.glob(os.path.join(consensus_dir, '*_consensus.fa')):
            cluster_name = os.path.basename(consensus).replace('_consensus.fa', '')
            with open(consensus) as infile:
                seq = infile.read().split('\n')[1]
                outfile.write(f">{cluster_name}\n{seq}\n")

def read_correction(consensus_dir, final_dir, threads, logger=None):
    """Polish consensus sequences using Racon"""
    if logger:
        logger.info('\nPolishing consensus sequences...')
    
    os.makedirs(final_dir, exist_ok=True)
    
    # Check if already processed
    final_seqs = os.path.join(final_dir, 'rep_seqs.fasta')
    if os.path.exists(final_seqs) and os.path.getsize(final_seqs) > 0:
        if logger:
            logger.info("Representative sequences already generated, skipping")
        return
    
    consensus_pooled = os.path.join(consensus_dir, 'consensus_pooled.fa')
    
    # Process each cluster
    for consensus in tqdm(glob.glob(os.path.join(consensus_dir, '*_consensus.fa')), 
                         desc="Polishing consensus sequences"):
        cluster_name = os.path.basename(consensus).replace('_consensus.fa', '')
        # Use the compressed fastq file
        cluster_fq = os.path.join(consensus_dir, f'{cluster_name}.fq.gz')
        
        # Mapping
        sam_file = os.path.join(final_dir, f'{cluster_name}.sam')
        cmd = f"minimap2 -ax map-ont -t {threads} {consensus} {cluster_fq} -o {sam_file}"
        run_command(cmd, logger)
        
        # Polish with Racon
        polished_file = os.path.join(final_dir, f'{cluster_name}_polished.fa')
        cmd = f"racon -m 8 -x -6 -g -8 -t {threads} {cluster_fq} {sam_file} {consensus} > {polished_file}"
        run_command(cmd, logger)
        
        # Clean up
        os.remove(sam_file)
    
    # Combine all polished sequences
    with open(final_seqs, 'w') as outfile:
        for polished in glob.glob(os.path.join(final_dir, '*_polished.fa')):
            cluster_name = os.path.basename(polished).replace('_polished.fa', '')
            with open(polished) as infile:
                seq = infile.read().split('\n')[1]
                outfile.write(f">{cluster_name}\n{seq}\n")
            os.remove(polished)

def main():
    parser = argparse.ArgumentParser(description='Fungal ITS Analysis Pipeline')
    
    # Required arguments
    parser.add_argument('--input_dir', required=True,
                       help='Directory containing input FASTQ files')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory')
    
    # Optional arguments
    parser.add_argument('--threads', type=int, default=mp.cpu_count(),
                       help='Number of threads (default: all available)')
    parser.add_argument('--kmer_length', type=int, default=6,
                       help='K-mer length (default: 6)')
    parser.add_argument('--cluster_size', type=int, default=25,
                       help='Minimum cluster size (default: 25)')
    parser.add_argument('--subsample', type=int, default=150,
                       help='Subsample threshold for large clusters (default: 60)')
    parser.add_argument('--select_epsilon', type=float, default=0.3,
                       help='Selection epsilon for HDBSCAN (default: 0.3)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(os.path.join(args.output_dir, 'logs'))
    logger.info("Starting Fungal ITS Analysis Pipeline")
    
    try:
        # Find input samples
        samples = [os.path.splitext(f)[0] for f in os.listdir(args.input_dir)
                  if f.endswith(('.fastq', '.fq', '.fastq.gz', '.fq.gz'))]
        
        if not samples:
            raise ValueError(f"No FASTQ files found in {args.input_dir}")
        
        logger.info(f"Found {len(samples)} samples: {', '.join(samples)}")
        
        # Create output directories
        kmer_dir = os.path.join(args.output_dir, 'kmers')
        cluster_dir = os.path.join(args.output_dir, 'clusters')
        shared_dir = os.path.join(args.output_dir, 'shared_clusters')
        consensus_dir = os.path.join(args.output_dir, 'consensus')
        final_dir = os.path.join(args.output_dir, 'polished')
        
        # Run pipeline stages
        # 1. K-mer counting
        kmer_counter(kmer_dir, args.input_dir, samples, args.threads, 
                    args.kmer_length, logger)
        
        # 2. Clustering
        cluster_sequences(kmer_dir, cluster_dir, samples, args.threads,
                         args.cluster_size, args.select_epsilon, logger)
        
        # 3. Share clusters between samples
        shared_clusters(cluster_dir, shared_dir, samples,
                       args.subsample, args.random_state, logger)
        
        # 4. Generate initial consensus
        fq_by_cluster(args.input_dir, shared_dir, consensus_dir, args.threads, logger)
        
        # 5. Final polishing
        read_correction(consensus_dir, final_dir, args.threads, logger)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
