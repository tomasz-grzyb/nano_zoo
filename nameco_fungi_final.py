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
                     select_epsilon=0.5, logger=None):
    """UMAP + HDBSCAN clustering optimized for fungal ITS but with bacterial-like processing"""
    if logger:
        logger.info("Starting UMAP + HDBSCAN clustering")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if already clustered
    all_clustered = True
    for sample in samples:
        cluster_file = os.path.join(output_dir, sample, 'clusters.tsv')
        if not os.path.exists(cluster_file):
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
                logger.info(f"Sample {sample} clusters exist, skipping")
            continue
            
        if logger:
            logger.info(f"Processing clustering for {sample} ({sample_idx}/{len(samples)})")
        
        # Read k-mer data
        kmer_file = os.path.join(kmer_dir, sample, 'kmers.tsv')
        data = pd.read_csv(kmer_file, sep='\t', index_col=0)
        
        # UMAP reduction with ITS parameters
        reducer = umap.UMAP(
            n_neighbors=25,
            min_dist=0.1,
            metric='braycurtis',
            n_components=2,
            random_state=42,
            n_jobs=threads
        )
        
        embedding = reducer.fit_transform(data.values)
        
        # HDBSCAN clustering with ITS parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=select_epsilon,
            cluster_selection_method='eom',
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(embedding)
        
        # Process clusters
        clusters = pd.DataFrame({'Feature': data.index, 'Cluster': labels})
        clusters = clusters.loc[clusters.Cluster >= 0]
        clusters.Cluster = 'Cluster_' + clusters.Cluster.astype(str)
        
        # Subsampling for cross-sample comparison (like in bacterial)
        for cid in clusters.Cluster.unique():
            sub = clusters.loc[clusters.Cluster == cid].copy()
            if len(sub) > 150:  # Changed from bacterial's 100 to ITS's 150
                sub = sub.sample(n=150, random_state=42)
            data.loc[sub.Feature.tolist(),'FullID'] = sample+'___'+cid+'___'
            
        data = data[data['FullID'].notna()]
        data.FullID = data.FullID + data.index.astype(str)
        data.set_index('FullID', inplace=True)
        
        # Save results
        data.to_csv(os.path.join(sample_dir, 'subsampled_ids.tsv'), sep='\t')
        clusters.to_csv(os.path.join(sample_dir, 'clusters.tsv'), sep='\t', index=False)
        
        if logger:
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"Found {n_clusters} clusters in {sample}")

def shared_clusters(cluster_dir, output_dir, samples, subsample_size=150, threads=12, logger=None):
    """Pool shared clusters between samples using bacterial-like logic but ITS parameters"""
    if logger:
        logger.info('\nStarting shared clusters pooling process...')
        logger.info(f'Found {len(samples)} samples to process')
    
    os.makedirs(output_dir, exist_ok=True)
    subsampled_dir = os.path.join(output_dir, 'Clusters_subsampled')
    os.makedirs(subsampled_dir, exist_ok=True)
    
    if logger:
        logger.info('Reading subsampled data from all samples...')
    
    # Read all subsampled data with progress tracking
    dfs = []
    for idx, sample in enumerate(samples, 1):
        if logger:
            logger.info(f'Processing sample {idx}/{len(samples)}: {sample}')
        df = pd.read_csv(os.path.join(cluster_dir, sample, 'subsampled_ids.tsv'), sep='\t', index_col=0)
        dfs.append(df)
        if logger:
            logger.info(f'Sample {sample} has {len(df)} features')
    data = pd.concat(dfs)
    
    if logger:
        logger.info(f'Total features across all samples: {len(data)}')
        logger.info('Starting UMAP reduction for shared clustering...')
    
    # UMAP reduction for shared clustering
    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.1,
        metric='braycurtis',
        n_jobs=threads,
        random_state=42
    )
    shared_embedding = reducer.fit_transform(data.values)
    
    if logger:
        logger.info('UMAP reduction completed')
        logger.info('Starting HDBSCAN clustering for shared clusters...')
    
    # HDBSCAN clustering for shared clusters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=150,
        cluster_selection_epsilon=0.5,
        cluster_selection_method='eom',
        core_dist_n_jobs=threads,
        prediction_data=True
    )
    labels = clusterer.fit_predict(shared_embedding)
    
    if logger:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f'HDBSCAN found {n_clusters} shared clusters')
    
    # Create DataFrame with features and cluster assignments
    shared = pd.DataFrame({
        'Feature': data.index.astype(str),
        'Cluster': labels
    })
    shared = shared[shared.Cluster >= 0]
    shared.Cluster = 'Cluster_' + shared.Cluster.astype(str)
    
    # Initialize counts and cluster dictionary
    if logger:
        logger.info('Initializing cluster processing...')
    
    counts = pd.DataFrame(columns=samples, index=shared.Cluster.unique())
    counts = counts.astype(float).fillna(0)
    clust_dict = {c:[] for c in shared.Cluster.unique()}
    i = len(clust_dict)-1
    
    # Process each sample
    for sample_idx, sample in enumerate(samples, 1):
        if logger:
            logger.info(f'Processing clusters for sample {sample_idx}/{len(samples)}: {sample}')
        
        unique = pd.read_csv(os.path.join(cluster_dir, sample, 'clusters.tsv'), sep='\t')
        n_clusters = len(unique.Cluster.unique())
        
        for cluster_idx, uclust in enumerate(unique.Cluster.unique(), 1):
            if logger and cluster_idx % 10 == 0:
                logger.info(f'  Progress: {cluster_idx}/{n_clusters} clusters ({(cluster_idx/n_clusters)*100:.1f}%)')
            
            uniq = unique.loc[unique.Cluster == uclust]
            pattern = f"{sample}___{uclust}___"
            # Zmieniona linia - używamy mask zamiast str.contains
            mask = shared['Feature'].apply(lambda x: pattern in str(x))
            shar = shared[mask]
            
            if len(shar) > 0:
                shar = shar.groupby('Cluster').size().reset_index(name='counts')
                shar = shar.sort_values('counts', ascending=False).reset_index()
                if shar.loc[0, 'counts'] > 40:
                    clust_dict[shar.loc[0, 'Cluster']] += uniq.Feature.tolist()
                    counts.loc[shar.loc[0, 'Cluster'], sample] += len(uniq)
                    continue
            
            i += 1
            counts.loc[f'Cluster_{i}', sample] = len(uniq)
            clust_dict[f'Cluster_{i}'] = uniq.Feature.tolist()
    
    if logger:
        logger.info('Finalizing results...')
    
    # Save results
    counts = counts.loc[~(counts==0).all(axis=1)]
    counts.index.name = 'Cluster'
    counts.to_csv(os.path.join(output_dir, 'cluster_counts.tsv'), sep='\t')
    
    # Save subsampled clusters
    total_clusters = len(clust_dict)
    for cluster_idx, (cluster, reads) in enumerate(clust_dict.items(), 1):
        if logger and cluster_idx % 50 == 0:
            logger.info(f'Saving cluster {cluster_idx}/{total_clusters} ({(cluster_idx/total_clusters)*100:.1f}%)')
        
        if len(reads) > subsample_size:
            reads = random.sample(reads, subsample_size)
        with open(os.path.join(subsampled_dir, f'{cluster}.txt'), 'w') as f:
            f.write('\n'.join(reads))
    
    if logger:
        logger.info(f'Completed processing {total_clusters} clusters')
        logger.info(f'Results saved to: {output_dir}')
        logger.info('Shared clusters processing completed successfully')

def fq_by_cluster(input_dir, shared_dir, consensus_dir, threads, logger=None):
    """Split fastq by clusters and generate initial consensus - bacterial logic with ITS parameters"""
    # Ten sam kod co wcześniej, ale z dodanym logowaniem
    if logger:
        logger.info('\nCreating fastq files for clusters and generating consensus...')
    
    os.makedirs(consensus_dir, exist_ok=True)
    
    # Check if consensus already generated
    consensus_pooled = os.path.join(consensus_dir, 'consensus_pooled.fa')
    if os.path.exists(consensus_pooled) and os.path.getsize(consensus_pooled) > 0:
        if logger:
            logger.info("Consensus sequences already generated, skipping")
        return

    # Reszta funkcji pozostaje bez zmian, bo logika przetwarzania fastq
    # jest taka sama dla bakterii i grzybów

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
    
    if not cluster_files:
        if logger:
            logger.error(f"No cluster files found in {os.path.join(shared_dir, 'Clusters_subsampled')}")
            logger.error("Directory contents: " + str(os.listdir(os.path.join(shared_dir, 'Clusters_subsampled'))))
        raise ValueError("No cluster files found for processing")
    
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
        
        # Compress cluster fastq
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
                       args.subsample, args.threads, logger)
        
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

