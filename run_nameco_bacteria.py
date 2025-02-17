#! /usr/bin/env python
import os
import re
import glob
import gzip
import time
import random
import argparse
import subprocess
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from importlib.metadata import version
from urllib.request import urlopen
from collections import Counter
from itertools import product
from Bio import SeqIO

#####################
##### FUNCTIONS #####
#####################

#Function to run bash commands
def bash(cmd):
    return subprocess.check_output(cmd, shell=True)
#Function to run bash commands with progress for long-running operations
def bash_with_progress(cmd, total_items=None, desc=None):
    """Run bash command with progress tracking for certain operations"""
    if desc:
        print(f"\n{desc}")
    
    if 'cat' in cmd and '*.f*q*' in cmd:
        # For file concatenation, count files first
        import glob
        files = glob.glob('*.f*q*')
        total = len(files)
        print(f"\rProcessing files: 0/{total} (0%)", end='', flush=True)
        
        # Modified command to process files one by one with progress
        for i, f in enumerate(files, 1):
            subprocess.check_output(f'cat {f} >> pooled.fq', shell=True)
            progress = (i / total) * 100
            print(f"\rProcessing files: {i}/{total} ({progress:.1f}%)", end='', flush=True)
        print()  # New line after completion
        return
    
    return subprocess.check_output(cmd, shell=True)

#Function to print greetings 
def greetings():
    grt = """
Thank you for using our tool!

######################################################
#                                                    #
# ##     #         ##     ##                         #                
# # #    #         # #   # #                         #
# # #    #         #  # #  #                         #
# #  #   #  #####  #   #   #   ###     ###     ###   #
# #  #   # #     # #       #  #   #   #   #   #   #  #
# #   #  #       # #       # #     # #       #     # #
# #   #  #  ###### #       # ####### #       #     # #
# #    # # #     # #       # #       #       #     # #
# #    # # #     # #       #  #   #   #   #   #   #  #
# #     ##  #####  #       #   ###     ###     ###   #
#                                                    #
######################################################

Written by Timur Yergaliyev
Powered by Coffee
If you used this pipeline, please cite our paper: XXX
Also, don't forget to cite the tools that were used in this pipeline
    """
    print(grt)
    

#Function to wrap messages into hashtags
def hashtags_wrapper(sub):
    print(f"\n{'#'*(len(sub)+8)}\n### {sub} ###\n{'#'*(len(sub)+8)}\n")


#Function to check logs
def log_checker(log, samples, file):
    skip, checks = [], []
    if os.path.exists(log):
        with open(log, 'rt') as txt:
            skip = [l.split(' ')[0] for l in txt.readlines() if l.endswith('done. Enjoy\n')]
    for sample in samples:
        checks.append(os.path.exists(file.format(sample)))
        checks.append(sample in skip)
    return skip, checks


#Function to run Chopper
def chopper(INPUT, SAMPLES, T, Q, MINL, MAXL, OUT, LOGS, log):
    print('Running chopper...')
    skip, checks = log_checker(log, SAMPLES, f'{OUT}/{{}}.fq.gz')
    if all(checks):
        return print(f'All samples were already chopped. Skipping')
    bash(f'mkdir -p {OUT} {LOGS}')
    for sample in SAMPLES:
        fq_out = f'{OUT}/{sample}.fq.gz'
        if os.path.exists(fq_out) and sample in skip:
            continue
        file = glob.glob(f"{INPUT}/{sample}.f*q*")[0]
        bash(f'echo "\n##### Processing {sample} #####\n" >> {log}')
        bash(f'echo "Chopping {sample}" >> {log}')
        pre = f'gunzip -c {file} -q |' if file.endswith('gz') else f'cat {file} |'
        bash(f'{pre} chopper -q {Q} -l {MINL} --maxlength {MAXL} -t {T} 2>> {log} | gzip > {fq_out}')
        bash(f'echo "{sample} done. Enjoy" >> {log}')


#Functions to count kmers of given length. 
#working function
def kmer_subcounter(kmers, rec, q):
    """Count k-mers for a single record"""
    count = [str(len(re.findall(f'(?={mer})', str(rec.seq)))) for mer in kmers]
    res = '\t'.join([rec.id] + count)
    q.put(res)

def kmer_writer(q, out, kmers, total_reads):
    """Write k-mer counts with progress tracking"""
    processed = 0
    last_update = time.time()
    update_interval = 5  # Update progress every 5 seconds
    
    with open(f'{out}/kmers.tsv', 'wt') as tab:
        tab.write('\t'.join(['ID'] + kmers) + '\n')
        while True:
            m = q.get()
            if m == 'kill':
                break
            tab.write(m + '\n')
            tab.flush()
            
            processed += 1
            current_time = time.time()
            if current_time - last_update >= update_interval:
                progress = (processed / total_reads) * 100
                print(f"\rProcessed {processed}/{total_reads} reads ({progress:.1f}%)", end='', flush=True)
                last_update = current_time

def count_reads(file):
    """Count total reads in FASTQ file"""
    if file.endswith('.gz'):
        with gzip.open(file, 'rt') as f:
            return sum(1 for line in f) // 4
    else:
        with open(file, 'rt') as f:
            return sum(1 for line in f) // 4

def kmer_counter(OUT, INPUT, SAMPLES, L, T, log):    
    print(f"Found samples: {SAMPLES}")
    print(f"Input directory: {INPUT}")
    print(f"Processing with {T} threads")
    
    skip, checks = log_checker(log, SAMPLES, f'{OUT}/{{}}/kmers.tsv')
    if all(checks):
        return print(f'Kmers were already counted. Skipping')
    
    # Pre-generate k-mers to avoid repeated computation
    nucleotides = 'ACGT'
    kmers = [''.join(c) for c in product(nucleotides, repeat=L)]
    print(f"Generated {len(kmers)} possible {L}-mers")
    
    # Process each sample
    for sample in SAMPLES:
        file = glob.glob(f"{INPUT}/{sample}.f*q*")[0]
        out = f'{OUT}/{sample}'
        bash(f'mkdir -p {out}')
        
        if os.path.exists(f'{out}/kmers.tsv') and sample in skip:
            continue
            
        # Count total reads for progress tracking
        print(f"\nCounting total reads in {sample}...")
        total_reads = count_reads(file)
        print(f"Found {total_reads} reads")
        
        # Setup multiprocessing
        q = mp.Manager().Queue()
        pool = mp.Pool(T)
        
        print(f"\nProcessing {sample}...")
        bash(f'echo "\n##### Processing {sample} #####\n" >> {log}')
        
        # Start writer process
        watcher = pool.apply_async(kmer_writer, (q, out, kmers, total_reads))
        
        # Process reads in chunks
        jobs = []
        chunk_size = 1000  # Process reads in chunks for better memory management
        current_chunk = []
        
        with gzip.open(file, 'rt') if file.endswith('.gz') else open(file, 'rt') as f:
            for rec in SeqIO.parse(f, 'fastq'):
                current_chunk.append(rec)
                if len(current_chunk) >= chunk_size:
                    for rec in current_chunk:
                        job = pool.apply_async(kmer_subcounter, (kmers, rec, q))
                        jobs.append(job)
                    current_chunk = []
                    
            # Process remaining reads
            for rec in current_chunk:
                job = pool.apply_async(kmer_subcounter, (kmers, rec, q))
                jobs.append(job)
                
        [job.get() for job in jobs]
        q.put('kill')
        pool.close()
        pool.join()
        
        print(f"\n{sample} completed")
        bash(f'echo "\n{sample} done. Enjoy" >> {log}')
        
    bash(f'echo "\nK-mers counted." >> {log}')

#Function to plot clusters
def plot_clusters(labels, clust_emb, sample, out):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    clustered = (labels >= 0)
    ax.scatter(clust_emb[~clustered, 0], clust_emb[~clustered, 1], 
               color=(0.5, 0.5, 0.5), s=3, alpha=0.5)
    ax.scatter(clust_emb[clustered, 0], clust_emb[clustered, 1], 
               c=labels[clustered], s=3, cmap='Spectral')
    plt.title(f'UMAP + HDBscan, {sample}')
    plt.savefig(out, dpi=300)
        
        
#Modified clustering function with progress tracking
def clustering_UMAP_HDBscan(OUT, SAMPLES, LOW, T, EPS, CLUST_SIZE, RSTAT, log):
    print('\nClustering sequences with UMAP and HDBscan...')
    print('"Noisy" (not assigned to any cluster) reads will be removed')
    
    skip, checks = log_checker(log, SAMPLES, f'{OUT}/{{}}/clusters.tsv')
    if all(checks) and os.path.exists(f'{OUT}/shared_clusters.tsv'):
        return print(f'Clusters for all samples were already created. Skipping')
    
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import umap
    
    total_samples = len(SAMPLES)
    for sample_idx, sample in enumerate(SAMPLES, 1):
        print(f"\rProcessing sample {sample_idx}/{total_samples} ({sample})", end='', flush=True)
        
        if os.path.exists(f'{OUT}/{sample}/clusters.tsv') and sample in skip:
            continue
            
        data = pd.read_csv(f'{OUT}/{sample}/kmers.tsv', sep='\t', index_col=0)
        print(f"\n - UMAP embedding for {sample}...")
        clust_emb = umap.UMAP(min_dist=0.1, n_jobs=T, low_memory=LOW, n_neighbors=25,
                              metric='braycurtis').fit_transform(data.values)
        
        print(f" - HDBSCAN clustering for {sample}...")
        labels = cluster.HDBSCAN(min_cluster_size=CLUST_SIZE, n_jobs=T, 
                                cluster_selection_epsilon=EPS,
                                cluster_selection_method='eom').fit_predict(clust_emb)
        
        # Rest of the processing...
        clusters = pd.DataFrame({'Feature': data.index, 'Cluster': labels})
        clusters = clusters.loc[clusters.Cluster >= 0]
        clusters.Cluster = 'Cluster_' + clusters.Cluster.astype(str)
        
        print(f" - Subsampling clusters for {sample}...")
        for cid in clusters.Cluster.unique():
            sub = clusters.loc[clusters.Cluster == cid].copy()
            if len(sub) > 100:
                sub = sub.sample(n=100, random_state=RSTAT)
            data.loc[sub.Feature.tolist(),'FullID'] = sample+'___'+cid+'___'
            
        data = data[data['FullID'].notna()]
        data.FullID = data.FullID + data.index.astype(str)
        data.set_index('FullID', inplace=True)
        data.to_csv(f'{OUT}/{sample}/subsampled_ids.tsv', sep='\t')
        clusters.to_csv(f'{OUT}/{sample}/clusters.tsv', sep='\t', index=False)
        plot_clusters(labels, clust_emb, sample, f'{OUT}/{sample}/clusters.png')
        bash(f'echo "{len(clusters)} features were clustered into {len(set(labels))} clusters" >> {log}')
    
    print("\n\nProcessing shared clusters...")
    dfs = []
    for sample in SAMPLES:
        df = pd.read_csv(f'{OUT}/{sample}/subsampled_ids.tsv', sep='\t', index_col=0)
        dfs.append(df)
    dfs = [df.apply(pd.to_numeric, downcast='integer') for df in dfs]
    data = pd.concat(dfs)
    
    print(" - UMAP embedding for shared clusters...")
    clust_emb = umap.UMAP(min_dist=0.1, n_jobs=T, low_memory=LOW, 
                          metric='braycurtis').fit_transform(data.values)
    
    print(" - HDBSCAN clustering for shared clusters...")
    labels = cluster.HDBSCAN(min_cluster_size=150, n_jobs=T, cluster_selection_epsilon=EPS,
             cluster_selection_method='eom').fit_predict(clust_emb)
    
    clusters = pd.DataFrame({'Feature': data.index, 'Cluster': labels})
    clusters = clusters.loc[clusters.Cluster >= 0]
    clusters.Cluster = 'Cluster_' + clusters.Cluster.astype(str)
    clusters.to_csv(f'{OUT}/shared_clusters.tsv', sep='\t', index=False)
    plot_clusters(labels, clust_emb, 'Shared clusters', f'{OUT}/clusters.png')
    bash(f'echo "Subsampled by cluster features were clustered between samples" >> {log}')
    print("\nClustering completed successfully!")
    
#Function to pool clusters from samples to shared clusters and recalculate abundances
def shared_clusters(OUT, FI, SAMPLES, RSTAT, SUBS, T, log):
    print('\nPooling shared clusters and recalculating abundances...')
    if os.path.exists(f'{FI}/cluster_counts.tsv'):
        return print(f'Clusters already pooled and recalculated. Skipping')
    shared = pd.read_csv(f'{OUT}/shared_clusters.tsv', sep='\t', index_col=0)
    counts = pd.DataFrame(columns=SAMPLES, index=shared.Cluster.unique())
    counts = counts.astype(float).fillna(0)
    clust_dict = {c:[] for c in shared.Cluster.unique()}
    i = len(clust_dict)-1
    
    for sample in SAMPLES:
        unique = pd.read_csv(f'{OUT}/{sample}/clusters.tsv', sep='\t', index_col=0)
        for uclust in unique.Cluster.unique():
            uniq = unique.loc[unique.Cluster == uclust]
            shar = shared.loc[shared.index.str.contains(f"{sample}___{uclust}___")]
            if len(shar) > 0:
                shar = shar.groupby('Cluster').size().reset_index(name='counts')
                shar = shar.sort_values('counts', ascending=False).reset_index()
                if shar.loc[0, 'counts'] > 40:
                    clust_dict[shar.loc[0, 'Cluster']] += uniq.index.tolist()
                    counts.loc[shar.loc[0, 'Cluster'], sample] += len(uniq)
                    continue
            i += 1
            counts.loc[f'Cluster_{i}', sample] = len(uniq)
            clust_dict.update({f'Cluster_{i}': uniq.index.tolist()})     
    counts = counts.astype(float).fillna(0)
    counts = counts.loc[~(counts==0).all(axis=1)]
    counts.index.name = 'Cluster'
    
    #write features by clusters
    print(f'Big clusters will be subsampled to {SUBS} reads for read correction!')
    random.seed(RSTAT)
    bash(f"mkdir -p {OUT}/Clusters_subsampled {FI}")
    with open(f"{OUT}/Clusters_subsampled/Pooled.txt", "w") as pooled:
        for k, v in clust_dict.items():
            pooled.write("{}\n{}\n".format(k, '\n'.join(v)))
            if len(v) > SUBS:
                v = random.sample(v, SUBS)
            with open(f"{OUT}/Clusters_subsampled/{k}.txt", "w") as clust:
                clust.write("{}".format('\n'.join(v)))
    counts.sort_index(key=lambda x: (x.to_series().str[8:].astype(int)), inplace=True)
    counts.to_csv(f'{FI}/cluster_counts.tsv', sep='\t')
    bash(f'echo "\nFeatures were stored by each shared cluster" >> {log}')
    print("\nFeatures were stored by each shared cluster")
    
    
#Functions to split fastq files by clusters and finding consensus for each cluster
#working function
def fq_splitter(out, cluster, log):
    fq = f'{out}/{cluster}.fq'
    fastq = f"{out}/../pooled.fq"
    bash(f"zgrep -f {out}/{cluster}.txt -F -A 3 {fastq} | grep -v '^--$' > {fq}")
    consensus = bash(f"spoa {fq}")
    with open(f'{out}/{cluster}_consensus.fa', 'wt') as cons:
        cons.write(">{}\n{}\n".format(cluster, str(consensus).split('\\n')[1]))
    bash(f'gzip -f {fq}')
    bash(f'echo "{cluster} done. Enjoy" >> {log}')
       
def fq_by_cluster(INPUT, subs, OUT, T, log):
    out = f'{OUT}/Clusters_subsampled'
    df = pd.read_csv(f"{OUT}/../Final_output/cluster_counts.tsv", sep='\t', index_col=0)
    clusters = df.index.tolist()
    file = f'{OUT}/consensus_pooled.fa'
    skip, checks = log_checker(log, clusters, f'{out}/{{}}_consensus.fa')
    if all(checks) and os.path.exists(file):
        if os.stat(file).st_size > 0:
            return print(f'Fastq files and consensuses for all clusters exists. Skipping')
    #pool
    print('\nCreating fastq files for subsampled clusters...')
    print(f'Big clusters will be subsampled to {subs} reads!')
    big = f"{OUT}/pooled.fq"
    if not os.path.exists(big):
        inp = glob.glob(f"{INPUT}/*.f*q*")
        if inp[0].endswith('.gz'):
            big = f"{OUT}/pooled.fq.gz"
        bash(f'cat {" ".join(inp)} > {big}')
        if big.endswith('.gz'):
            bash(f'gunzip {big}')
    #must use Manager queue here, or will not work
    pool = mp.Pool(T)
    
    #fire off workers
    jobs = []
    for cluster in clusters:
        if os.path.exists(f'{out}/{cluster}.fq') and os.path.exists(f'{out}/{cluster}_consensus.fa'):
            continue
        job = pool.apply_async(fq_splitter, (out, cluster, log))
        jobs.append(job)
    [job.get() for job in jobs]
    
    #pool consensuses
    with open(file, 'w') as pooled:
        for cluster in clusters:
            with open(f'{out}/{cluster}_consensus.fa', 'r') as cons:
                pooled.write(cons.read())
                

# Add progress tracking to read correction
def read_correction(T, OUT, FI, log):
    print(f'Polishing with Racon...')
    bash(f'mkdir -p {OUT}')
    file = f'{OUT}/../Clustering/consensus_pooled.fa'
    corr = f'{FI}/rep_seqs.fasta'
    df = pd.read_csv(f"{OUT}/../Final_output/cluster_counts.tsv", sep='\t', index_col=0)
    clusters = df.index.tolist()
    total_clusters = len(clusters)
    
    skip, checks = log_checker(log, ['mock'], f'{OUT}/{{}}/mock')
    if os.path.exists(corr):
        if os.stat(corr).st_size > 0 and os.stat(file).st_size > 0:
            if int(bash(f'grep -c "^>" {file}')) == int(bash(f'grep -c "^>" {corr}')):
                return print(f'Consensuses for all clusters were already corrected. Skipping')
    
    for idx, cluster in enumerate(clusters, 1):
        progress = (idx / total_clusters) * 100
        print(f"\rProcessing cluster {idx}/{total_clusters} ({progress:.1f}%)", end='', flush=True)
        
        po = f"{OUT}/{cluster}_racon.fa"
        if os.path.exists(po) and cluster in skip:
            continue
            
        fa = f'{OUT}/../Clustering/Clusters_subsampled/{cluster}_consensus.fa'
        sam = f"{OUT}/{cluster}.sam"
        fq = f'{OUT}/../Clustering/Clusters_subsampled/{cluster}.fq.gz'
        
        # Mapping
        bash(f"minimap2 -ax map-ont -t {T} {fa} {fq} -o {sam} 2>> {log}")
        
        # Polishing
        bash(f'racon -m 8 -x -6 -g -8 -t {T} {fq} {sam} {fa} > {po} 2>> {log}')
        bash(f'rm {sam}')
    
    print("\nCollecting corrected sequences...")
    with open(corr, "w") as corrected:
        for cluster in clusters:
            with open(f"{OUT}/{cluster}_racon.fa", "rt") as rep:
                corrected.write(">{id}\n{seq}\n".format(id=cluster, seq=rep.read().split('\n')[1]))
    
    print("\nRead correction completed!")


#Functions for taxonomy annotation with Blast and NCBI
def ncbi_parser(blast, taxid, q):
    time.sleep(5) #bypass NCBI restriction
    url = 'https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?mode=Info&id={ID}'
    ranks = {
        'd': ['"superkingdom">', '<'],
        'p': ['"phylum">', '<'],
        'c': ['"class">', '<'],
        'o': ['"order">', '<'],
        'f': ['"family">', '<'],
        'g': ['"genus">', '<'],
        's': ['Taxonomy browser (', ')']}
    
    ID = str(taxid).split('.')[0]
    page = urlopen(url.format(ID=ID), timeout=360)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    taxonomy = [taxid]
    for rank, (start, end) in ranks.items():
        taxonomy.append(f'{rank}__' + html.split(start)[-1].split(end)[0])
    last = taxonomy[-1]
    while len(taxonomy) < 7:
        taxonomy.append(f'{taxonomy[-1]}')
    taxonomy[-1] = ' '.join(taxonomy[-1].split(' ')[:2]).replace('incertae', 'incertae sedis')
    return q.put(taxonomy)

#apply thresholds based on percent identity to mask false positive annotations
def taxonomy_thresholds(bclust, thresholds):
    for ind in bclust.index:
        taxon = bclust.loc[ind, 'Taxon']
        last = ''
        for rank, perc in thresholds.items():
            prefix = f"{rank[0].lower()}__"
            pat = taxon.split(prefix)[-1].split(';')[0]
            if bclust.loc[ind, 'pind'] >= perc:
                last = pat
            if bclust.loc[ind, 'pind'] < perc:
                taxon = taxon.replace(prefix+pat, f"{prefix}{last} unclassified")
                bclust.loc[ind, 'Taxon'] = taxon
    return(bclust)

#select top-hit based on consensus taxonomy
def top_hit(bclust, taxa, frac):
    if len(bclust) == 0:
        return 'Unclassified', 0.0
    
    taxa_counts = bclust["Taxon"].value_counts()
    bclust["Taxa_counts"] = bclust["Taxon"].map(taxa_counts)
    bclust.sort_values(["Taxa_counts", 'bitscore', 'pind'], ascending=[False, False, False], inplace=True)
    
    taxon, pind = bclust.Taxon.iloc[0], bclust.pind.iloc[0]
    if len(bclust.loc[bclust.Taxa_counts==bclust.Taxa_counts.max()])/len(bclust) < frac:
        taxon = taxon.rsplit(';',1)[0] + taxon.rsplit(';',1)[-1].split(' ')[0] + ' unclassified'
    
    return taxon, pind

def taxonomy_annotation(DB, gap, frac, T, OUT, FI, DBpath, log):
    print(f'Starting taxonomy annotations with blastn against {DB}...')
    Q=f'{FI}/rep_seqs.fasta'
    DBpath = DBpath.format(OUT=OUT, DB=DB)
    queries = [l[1:].split(' ')[0].split('\n')[0] for l in open(Q, 'rt') if l.startswith('>')]
    total_queries = len(queries)
    
    thresholds = {'Domain': 65, 'Phylum': 75, 'Class': 78.5,
                  'Order': 82, 'Family': 86.5, 'Genus': 94.5, 'Species': 98}
    taxa = pd.DataFrame(columns=['Taxon', 'Perc. id.'])
    bash(f'mkdir -p {OUT} {FI}')
    
    if DB == 'NCBI':
        if not os.path.exists(f"{DBpath}/16S_ribosomal_RNA.ndb"):
            print(f'Creating database...')
            seq = 'https://ftp.ncbi.nlm.nih.gov/blast/db/16S_ribosomal_RNA.tar.gz'
            print("Downloading NCBI database...")
            bash(f'wget -P {DBpath} {seq} 2>> {log} --progress=bar:force')
            print("Extracting database...")
            bash_with_progress(f'tar -xzvf {DBpath}/16S_ribosomal_RNA.tar.gz -C {DBpath}/ 2>> {log}',
                             desc="Extracting database files...")
            bash(f'rm {DBpath}/*.tar.gz')
        
        if not os.path.exists(f"{OUT}/{DB}-blastn.tsv"):
            print(f'\nRunning BLAST against NCBI database...')
            bash(f'blastn -query {Q} -db {DBpath}/16S_ribosomal_RNA -task blastn \
                   -num_threads {T} -out {OUT}/{DB}-blastn.tsv -max_target_seqs 50 -max_hsps 50 \
                   -outfmt "6 qseqid staxids evalue length pident nident bitscore score gaps" 2>> {log}')
        
        if not os.path.exists(f"{FI}/{DB}-taxonomy.tsv"):
            print('\nParsing NCBI taxonomies...')
            blast = pd.read_csv(f"{OUT}/{DB}-blastn.tsv", sep='\t', header=None, 
                    names=['Cluster', 'taxid', 'eval', 'length', 'pind', 'nind', 'bitscore', 'score', 'gaps'])
            blast = blast.sort_values(['bitscore', 'eval'], ascending=[False, False])

            for idx, cluster in enumerate(queries, 1):
                progress = (idx / total_queries) * 100
                print(f"\rProcessing taxonomy {idx}/{total_queries} ({progress:.1f}%)", end='', flush=True)
                
                bclust = blast.loc[blast.Cluster == cluster].copy()
                bclust = bclust.loc[bclust.bitscore >= bclust.bitscore.max() - gap]
                
                q = mp.Manager().Queue()    
                pool = mp.Pool(2)
                jobs = []
                for taxid in bclust.taxid.unique():
                    job = pool.apply_async(ncbi_parser, (bclust, taxid, q))
                    jobs.append(job)
                [job.get() for job in jobs]
                
                while not q.empty():
                    qout = q.get()
                    taxid, taxonomy = qout[0], qout[1:]
                    bclust.loc[bclust.taxid == taxid, 'Taxon'] = ';'.join(taxonomy)
                bclust = taxonomy_thresholds(bclust, thresholds)
                taxa.loc[cluster, ['Taxon', 'Perc. id.']] = top_hit(bclust, taxa, frac)
                
    if DB == 'GTDB':
        if not os.path.exists(f"{DBpath}/ssu_all.fna.ndb"):
            print(f'Creating database...')
            seq = 'https://data.ace.uq.edu.au/public/gtdb/data/releases/latest/genomic_files_all/ssu_all.fna.gz'
            bash(f'mkdir -p {DBpath}')
            bash(f'wget -P {DBpath} {seq} 2>> {log}')
            bash(f'gunzip {DBpath}/ssu_all.fna.gz 2>> {log}')
            bash(f'makeblastdb -in {DBpath}/ssu_all.fna -title "ssu_all.fna" \
                   -parse_seqids -dbtype "nucl"')
            
            with open(f'{DBpath}/ssu_all.fna', 'rt') as fa:
                ls = [l[1:].replace(' d_','\td_').split(' [')[0] for l in fa.readlines() if l.startswith('>')]
                with open(f'{DBpath}/map.tsv', 'wt') as ref:
                    ref.write('SeqID\tTaxonomy\n')
                    ref.write('\n'.join(ls))
            bash(f'rm {DBpath}/ssu_all.fna')
        else:
            print('Database exists. Skipping')
            bash(f'echo "{DB} database exists. Skipping." >> {log}')   
        
        if not os.path.exists(f"{OUT}/{DB}-blastn.tsv"):
            print(f'\nAssigning taxonomy...')
            bash(f'blastn -query {Q} -db {DBpath}/ssu_all.fna -task blastn \
                   -num_threads {T} -out {OUT}/{DB}-blastn.tsv -max_target_seqs 50 -max_hsps 50 \
                   -outfmt "6 qseqid sseqid evalue length pident nident bitscore score gaps" 2>> {log}')
        else:
            print('\nBlastn output exists. Skipping')
            bash(f'echo "Blastn output exists. Skipping." >> {log}')

        blast = pd.read_csv(f"{OUT}/{DB}-blastn.tsv", sep='\t', header=None, 
                names=['Cluster', 'SeqID', 'eval', 'length', 'pind', 'nind', 'bitscore', 'score', 'gaps'])
        blast = blast.sort_values(['bitscore', 'eval'], ascending=[False, False])

        if not os.path.exists(f"{FI}/{DB}-taxonomy.tsv"):
            print('\nMapping GTDB to get full taxonomies...')
            mapp = pd.read_csv(f'{DBpath}/map.tsv', sep='\t')
            mapp.Taxonomy = mapp.Taxonomy.apply(lambda x: x.rsplit(';', 1)[0] +';'+ 
                            ' '.join(x.rsplit(';', 1)[-1].replace('_', ' ').replace('  ', '__').split(' ')[:2]))
            mapping = dict(mapp[['SeqID', 'Taxonomy']].values)
            for cluster in queries:
                bclust = blast.loc[blast.Cluster == cluster].copy()
                bclust = bclust.loc[bclust.bitscore > bclust.bitscore.max() - gap]
                bclust['Taxon'] = bclust['SeqID'].map(mapping)
                bclust = taxonomy_thresholds(bclust, thresholds)
                taxa.loc[cluster, ['Taxon', 'Perc. id.']] = top_hit(bclust, taxa, frac)  
        else:
            print('\nTaxonomy exists. Skipping')
            bash(f'echo "Taxonomy exists. Skipping." >> {log}')  

    if len(taxa) != 0:
        for cluster in queries:
            if cluster not in blast.Cluster.tolist():
                taxa.loc[cluster, 'Taxon'] = 'Unclassified'
        taxa.index.rename('Feature ID', inplace=True)
        taxa.to_csv(f'{FI}/{DB}-taxonomy-q2.tsv', sep='\t')
        for rank in thresholds:
            taxa[rank] = taxa.Taxon.apply(lambda x: x.split(f"{rank[0].lower()}__")[-1].split(';')[0])
        taxa.drop('Taxon', axis=1, inplace=True)
        taxa.index.rename('Cluster', inplace=True)
        taxa.to_csv(f'{FI}/{DB}-taxonomy.tsv', sep='\t')
        
    print('\nChecking if collapsed taxonomies exist...')
    taxa = pd.read_csv(f'{FI}/{DB}-taxonomy.tsv', sep='\t', index_col=0)
    counts = pd.read_csv(f'{FI}/cluster_counts.tsv', sep='\t', index_col=0)
    for rank in thresholds:
        if os.path.exists(f"{FI}/{DB}-taxonomy-{rank}.tsv"):
            continue
        print(f'Collapsing to {rank}')
        coll = counts.copy()
        coll[rank] = taxa[rank]
        coll = coll.groupby(rank).sum()
        coll.to_csv(f"{FI}/{DB}-taxonomy-{rank}.tsv", sep='\t')
    
    print("\nTaxonomy annotation completed!")
        
#Function to run NaMeco 
def main():
    ############
    # ARGPARSE #
    ############
    inp_dir_help = " ".join(['Path to the folder with reads, absolute or relative.', 
                             'Reads should be in the fastq or fq format, gziped or not'])
    out_dir_help = " ".join(['Path to the directory to store output files, absolute or relative.', 
                             'If not provided, folder "Nameco_out" will be created in working directory'])
    subsample_help = " ".join(['Subsample bigger than that threshold clusters for consensus creation and', 
                               'polishing by Racon (default 500)'])
    gap_help = " ".join(['Gap between the bit score of the best hit and others,',
                        'that are considered with the top hit for taxonomy selection (default 1)'])
    frac_help = " ".join(['If numerous hits retained after gap filtering, consensus taxon should have at least this',
                'fraction to be selected. Otherwise it will be set to lower level + unclassified (default 0.6)'])
    database_help = " ".join(['Database for taxonomy assignment (default GTDB).', 
                              'Only GTDB or NCBI are currently supported'])
    db_path_help = " ".join(['Path to store/existing database (default $out_dir/$database).', 
                             'Please use only databases, created by previous NaMeco run to avoid errors'])

    parser = argparse.ArgumentParser(prog='nameco')
    parser._action_groups.pop()
    req = parser.add_argument_group('required arguments')
    opt = parser.add_argument_group('optional arguments')
    req.add_argument("--inp_dir", help=inp_dir_help, required=True)
    opt.add_argument("--out_dir", help=out_dir_help, default='NaMeco_out')
    opt.add_argument("--threads", help="The number of threads/cpus (default 2)", type=int, default=2)
    opt.add_argument("--qc", help="Run chopper for quality control (default)", action='store_true', default=True)
    opt.add_argument("--no-qc", help="Skip chopper for quality control", dest='qc', action='store_false')
    opt.add_argument("--phred", help="Minimum phred average score for chopper (default 8)", type=int, default=8)
    opt.add_argument("--min_length", help="Minimum read length for chopper (default 1300)", type=int, default=1300)
    opt.add_argument("--max_length", help="Maximum read length for chopper (default 1700)", type=int, default=1700)
    opt.add_argument("--kmer", help="K-mer length for clustering (default 5)", type=int, default=5)
    opt.add_argument("--no-low", help="Don't restrict RAM for UMAP (default)", action='store_false', default=False)
    opt.add_argument("--low", help="Reduce RAM usage by UMAP", dest='no_low', action='store_true',)
    opt.add_argument("--cluster_size", help="Minimum cluster size for HDBscan (default 500, not < 100!)", type=int, default=500)
    opt.add_argument("--subsample", help=subsample_help, type=int, default=500)
    opt.add_argument("--select_epsilon", help="Selection epsilon for clusters (default 0.5)", type=float, default=0.5)
    opt.add_argument("--gap", help=gap_help, type=float, default=1)
    opt.add_argument("--min_fraction", help=frac_help, type=float, default=.6)
    opt.add_argument("--random_state", help="Random state for subsampling (default 42)", type=int, default=42)
    opt.add_argument('--database', default='GTDB', choices=['GTDB', 'NCBI'], help=database_help)
    opt.add_argument('--db_path', help=db_path_help, default='{OUT}/{DB}')
    opt.add_argument('--version', help="Check the version", action="version", version=version("nameco"))
    args = parser.parse_args()
    if args.cluster_size < 100:
        raise ValueError('Minimum cluster size can not be less than 100')
    
    greetings()
    INPDIR = args.inp_dir
    LOGS = f'{args.out_dir}/Logs'
    QC = f'{args.out_dir}/Quality_control'
    CL = f'{args.out_dir}/Clustering'
    FS = f'{args.out_dir}/FastANI_selection'
    RC = f'{args.out_dir}/Read_correction'
    TA = f'{args.out_dir}/Taxonomy_annotation'
    FI = f'{args.out_dir}/Final_output'
    
    exts = (".fastq.gz", ".fq.gz", ".fastq", ".fq")
    SAMPLES = [f.split('.')[0] for f in os.listdir(INPDIR) if f.endswith(exts)]
    print('Only "*.fastq.gz", "*.fq.gz", "*.fastq" or "*.fq" files will be procsessed')
    if len(SAMPLES) == 0:
        raise ValueError('Input directory does not contain fastq.gz or fq.gz files')
        
    ###################
    # Quality control #
    ###################
    module = QC.split('/')[-1]
    hashtags_wrapper(f"{module.replace('_', ' ')} module")
    log = f"{LOGS}/{module}.log"
    if args.qc:
        chopper(INPUT=INPDIR, SAMPLES=SAMPLES, T=args.threads, LOGS=LOGS, log=log, Q=args.phred, 
                MINL=args.min_length, MAXL=args.max_length, OUT=f'{QC}/Chopper')
        INPDIR = f'{QC}/Chopper'
        print('\nPlease, cite chopper: https://doi.org/10.1093/bioinformatics/btad311')
    else:
        print(f"{module.replace('_', ' ')} module disabled. Skipping")
    print(f"\nEnd of the {module.replace('_', ' ')} module")
    
    ##############
    # Clustering #
    ##############
    module = CL.split('/')[-1]
    hashtags_wrapper(f"{module.replace('_', ' ')} module")
    log = f"{LOGS}/{module}.log"
    #kmers counting
    print(f"Counting kmers ({args.kmer}-mers) for all samples...")
    kmer_counter(OUT=CL, INPUT=INPDIR, SAMPLES=SAMPLES, T=args.threads, L=args.kmer, log=log)
    #clustering with UMAP + HDBscan
    clustering_UMAP_HDBscan(OUT=CL, T=args.threads, CLUST_SIZE=args.cluster_size, EPS=args.select_epsilon,
                            SAMPLES=SAMPLES, LOW=args.no_low, RSTAT=args.random_state, log=log)
    #pool clusters from samples to shared clusters and recalculate abundances
    shared_clusters(OUT=CL, FI=FI, SAMPLES=SAMPLES, RSTAT=args.random_state, 
                    SUBS=args.subsample, T=args.threads, log=log)
    #spliting fasta by cluster
    fq_by_cluster(INPUT=INPDIR, subs=args.subsample, OUT=CL, T=args.threads, log=log)
    print('\nPlease, cite UMAP: https://doi.org/10.21105/joss.00861')
    print('Please, cite HDBscan: https://doi.org/10.21105/joss.00205')
    print('Please, cite SPOA: https://doi.org/10.1101%2Fgr.214270.116')
    print(f"\nEnd of the {module.replace('_', ' ')} module")
    
    ###################
    # Read correction #
    ###################
    module = RC.split('/')[-1]
    hashtags_wrapper(f"{module.replace('_', ' ')} module")
    log = f"{LOGS}/{module}.log"
    read_correction(OUT=RC, FI=FI, T=args.threads, log=log)
    print('\nPlease, cite minimap2: https://doi.org/10.1093/bioinformatics/bty191')
    print('Please, cite racon: https://doi.org/10.1101%2Fgr.214270.116')
    print(f"\nEnd of the {module.replace('_', ' ')} module")

    #######################
    # Taxonomy annotation #
    #######################
    module = TA.split('/')[-1]
    hashtags_wrapper(f"{module.replace('_', ' ')} module")
    log = f"{LOGS}/{module}.log"
    taxonomy_annotation(DB=args.database, gap=args.gap, frac=args.min_fraction,  T=args.threads, 
                        OUT=TA, FI=FI, DBpath=args.db_path, log=log)
    if args.database == 'GTDB':
        print('\nPlease, cite GTDB database: https://doi.org/10.1038/s41587-020-0501-8')
    if args.database == 'NCBI':
        print('\nPlease, cite NCBI database: https://doi.org/10.1093/nar/gkab1112')
    print('\nPlease, cite BLAST: https://doi.org/10.1016/s0022-2836(05)80360-2')
    print(f"\nEnd of the {module.replace('_', ' ')} module")
    module = "NaMeco run successfully completed. Enjoy your data!"
    hashtags_wrapper(f"{module.replace('_', ' ')}")

if __name__ == '__main__':
    main()
