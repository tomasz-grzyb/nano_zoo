#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
import subprocess
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
import gzip
import multiprocessing as mp

def setup_logging(output_dir):
    """Set up logging with file and console output"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'unite_taxonomy.log')
    
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
        if logger and result.stderr:
            logger.warning(f"Command stderr: {result.stderr.strip()}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"Command failed: {cmd}")
            logger.error(f"Error output: {e.stderr}")
        raise

def prepare_unite_database(unite_gz, output_dir, threads, logger=None):
    """Prepare UNITE database for BLAST searches"""
    if logger:
        logger.info("Preparing UNITE database...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    unite_fasta = os.path.join(output_dir, "unite_refs.fasta")
    blast_db = os.path.join(output_dir, "unite_blast_db")
    taxonomy_map = os.path.join(output_dir, "unite_taxonomy.map")
    
    if os.path.exists(f"{blast_db}.nhr"):
        if logger:
            logger.info("BLAST database already exists, skipping preparation")
        return blast_db, taxonomy_map
    
    if logger:
        logger.info("Processing UNITE sequences...")
    
    records_processed = 0
    with gzip.open(unite_gz, 'rt') as gz, open(unite_fasta, 'w') as fasta, open(taxonomy_map, 'w') as tax:
        tax.write("sequence_id\tkingdom\tphylum\tclass\torder\tfamily\tgenus\tspecies\n")
        
        for record in SeqIO.parse(gz, 'fasta'):
            # Parse UNITE header format: UDBXXXXXX|taxonomy|SHXXXXXX.XXFU
            header_parts = record.description.split('|')
            if len(header_parts) < 2:  # Need at least ID and taxonomy
                continue
                
            unite_id = header_parts[0]
            taxonomy = header_parts[1].split(';')
            sh_code = header_parts[2] if len(header_parts) > 2 else "NOSH"
            
            # Parse taxonomy
            tax_dict = {}
            for rank in taxonomy:
                if rank.startswith('k__'): tax_dict['kingdom'] = rank[3:]
                elif rank.startswith('p__'): tax_dict['phylum'] = rank[3:]
                elif rank.startswith('c__'): tax_dict['class'] = rank[3:]
                elif rank.startswith('o__'): tax_dict['order'] = rank[3:]
                elif rank.startswith('f__'): tax_dict['family'] = rank[3:]
                elif rank.startswith('g__'): tax_dict['genus'] = rank[3:]
                elif rank.startswith('s__'): tax_dict['species'] = rank[3:]
            
            # Write FASTA with simplified header
            fasta.write(f">{unite_id}|{sh_code}\n{str(record.seq)}\n")
            
            # Write taxonomy mapping
            tax_line = [
                unite_id,
                tax_dict.get('kingdom', 'Fungi'),
                tax_dict.get('phylum', 'unclassified'),
                tax_dict.get('class', 'unclassified'),
                tax_dict.get('order', 'unclassified'),
                tax_dict.get('family', 'unclassified'),
                tax_dict.get('genus', 'unclassified'),
                tax_dict.get('species', 'unclassified').replace('_', ' ')
            ]
            tax.write('\t'.join(tax_line) + '\n')
            
            records_processed += 1
            if records_processed % 1000 == 0 and logger:
                logger.info(f"Processed {records_processed} sequences")
    
    if logger:
        logger.info(f"Total sequences processed: {records_processed}")
        logger.info("Creating BLAST database...")
    
    cmd = f"makeblastdb -in {unite_fasta} -dbtype nucl -out {blast_db} -parse_seqids"
    run_command(cmd, logger)
    
    os.remove(unite_fasta)
    return blast_db, taxonomy_map

def taxonomy_annotation_unite(rep_seqs, unite_db, taxonomy_map, output_dir, threads, logger=None):
    """Annotate taxonomy using UNITE database with thresholds specific for fungi"""
    if logger:
        logger.info("Starting taxonomy annotation with UNITE...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Thresholds for Nanopore ITS sequences
    thresholds = {
        'Kingdom': 65,
        'Phylum': 70,
        'Class': 75,
        'Order': 80,
        'Family': 85,
        'Genus' : 90,
        'Species': 95
    }
    
    # Run BLAST
    blast_out = os.path.join(output_dir, "unite_blast.tsv")
    if not os.path.exists(blast_out):
        cmd = f"blastn -query {rep_seqs} -db {unite_db} -task blastn \
               -num_threads {threads} -out {blast_out} \
               -max_target_seqs 50 -max_hsps 1 \
               -outfmt '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore'"
        run_command(cmd, logger)
    
    # Process BLAST results
    blast_df = pd.read_csv(blast_out, sep='\t',
                          names=['query_id', 'subject_id', 'pident', 'length',
                                'mismatch', 'gapopen', 'qstart', 'qend', 'sstart',
                                'send', 'evalue', 'bitscore'])
    
    # Load taxonomy mapping
    tax_df = pd.read_csv(taxonomy_map, sep='\t')
    tax_dict = dict(zip(tax_df['sequence_id'], 
                       tax_df.apply(lambda x: dict(zip(['kingdom', 'phylum', 'class', 'order', 
                                                      'family', 'genus', 'species'], x[1:])), axis=1)))
    
    # Process results for each query
    results = []
    for query in blast_df['query_id'].unique():
        hits = blast_df[blast_df['query_id'] == query].sort_values('bitscore', ascending=False)
        best_hit = hits.iloc[0]
        
        # Get sequence ID from BLAST hit
        seq_id = best_hit['subject_id'].split('|')[0]
        taxonomy = tax_dict.get(seq_id, {})
        
        # Apply thresholds
        tax_result = {'query_id': query, 'percent_identity': best_hit['pident']}
        for rank, threshold in thresholds.items():
            rank_lower = rank.lower()
            if best_hit['pident'] >= threshold:
                tax_result[rank] = taxonomy.get(rank_lower, 'unclassified')
            else:
                tax_result[rank] = 'unclassified'
        
        results.append(tax_result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "taxonomy_annotations.tsv"), sep='\t', index=False)
    
    if logger:
        logger.info(f"Processed {len(results)} sequences")
        logger.info(f"Results saved to {os.path.join(output_dir, 'taxonomy_annotations.tsv')}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='UNITE Taxonomy Assignment for NaMeCo')
    
    parser.add_argument('--input_fasta', required=True,
                      help='Path to input FASTA file (rep_seqs.fasta from NaMeCo)')
    parser.add_argument('--unite_db', required=True,
                      help='Path to UNITE database (FASTA.GZ file)')
    parser.add_argument('--output_dir', required=True,
                      help='Output directory')
    parser.add_argument('--threads', type=int, default=mp.cpu_count(),
                      help='Number of CPU threads to use')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.output_dir)
    
    try:
        logger.info("Starting UNITE taxonomy pipeline")
        
        # Prepare database
        blast_db, taxonomy_map = prepare_unite_database(
            args.unite_db, 
            os.path.join(args.output_dir, 'unite_db'),
            args.threads,
            logger
        )
        
        # Run taxonomy annotation
        taxonomy_annotation_unite(
            args.input_fasta,
            blast_db,
            taxonomy_map,
            args.output_dir,
            args.threads,
            logger
        )
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

