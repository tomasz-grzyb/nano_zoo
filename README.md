# NanoZoo: NaMeCo-based Pipelines for Bacterial and Fungal Sequence Analysis

## Acknowledgments
This project builds directly upon the NaMeCo pipeline created by Timur Yergaliyev (https://github.com/timyerg/NaMeco/). We are deeply grateful for their fundamental contribution to the field of Nanopore sequence analysis. If you use these pipelines, please cite:

- Original NaMeCo: [Citation to be added]
- Our modifications: [Citation to be added]

For transparency and proper acknowledgment of resources used: the code implementation was significantly supported by Claude (Anthropic AI), however, all scientific decisions, pipeline architecture choices, and parameter optimizations were developed and validated independently by our research team.

## Project Background
NanoZoo originated from a microbiome analysis project at Łódź Zoo in Poland, where we needed to process both bacterial and fungal sequences from air samples. This unique requirement led us to modify and extend the original NaMeCo pipeline to handle both sequence types effectively.

## Overview
NanoZoo consists of two specialized pipelines:

### Fungal Pipeline (New)
Our main contribution is a complete pipeline for fungal ITS sequence analysis. It was developed to address the specific challenges of fungal sequences, which differ significantly from bacterial ones in terms of sequence variability and database requirements. Key features include:

- Full processing pipeline optimized for fungal ITS regions
- Integration with the UNITE database for accurate fungal taxonomy assignment
- Modified clustering approach suitable for fungal sequence characteristics
- Specialized two-component system with separate taxonomy module
- Custom parameter sets optimized for fungal data

### Bacterial Pipeline (Modified)
We also refined the original NaMeCo bacterial pipeline to enhance its usability and monitoring capabilities:

- Improved progress tracking for long-running processes
- Enhanced memory management for large datasets
- Better error handling and recovery options
- Maintained core functionality of original NaMeCo

## Key Applications
- Analysis of environmental samples containing both bacterial and fungal sequences
- Processing of air microbiome samples
- Large-scale Nanopore sequencing projects requiring robust processing
- Projects needing specialized fungal taxonomy assignment

## Technical Highlights
1. Two independent yet compatible pipelines
2. Enhanced monitoring and error handling
3. Robust memory management for large datasets
4. Extended database support with UNITE integration
5. Comprehensive logging system
6. Detailed documentation and usage guidelines

## When to Use NanoZoo
Consider using NanoZoo when you:
- Need to analyze fungal ITS sequences from Nanopore data
- Process both bacterial and fungal sequences from the same project
- Require robust progress monitoring and error handling
- Work with environmental samples containing diverse microbial communities

## Installation and Usage
[Detailed installation instructions to be added]

Our modifications maintain compatibility with the original NaMeCo workflow while providing specialized solutions for different sequence types. The pipeline has been extensively tested on air microbiome samples from zoo environments, making it particularly suitable for environmental monitoring projects.
