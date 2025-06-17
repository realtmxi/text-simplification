# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import faiss
from tqdm import tqdm
import sys
import os
import numpy as np
import math
import time
import json
import hashlib
import datetime
import concurrent.futures

import glob
import pandas as pd

from muss.utils.submitit import get_executor
from muss.utils.helpers import get_file_hash, get_files_hash, log_action, yield_lines
from muss.resources.paths import get_dataset_dir
from muss.mining.preprocessing import (
    create_base_index,
    get_index_name,
)
from muss.mining.nn_search import (
    get_cache_dir,
    get_results_path,
    compute_and_save_nn_batched,
    compute_and_save_nn_batched_parallel,
    get_paraphrase_pairs,
    get_pairs_path,
    compute_and_save_simplification_pairs,
    get_index_path,
    compute_and_save_embeddings,
    get_filter_string_representation,
    combine_simplifications_in_dataset,
    get_simplification_pairs_paths,
)
from muss.mining.filtering import SimplicityScorer

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # muss directory
LASER_DIR = PROJECT_ROOT / 'resources/tools/LASER'
NLLB_DIR = LASER_DIR / 'nllb'
CC100_PATH = Path('/data/models/muxin/CC100-sinhala/si.txt')
MADLAD_DIR = Path('/data/models/muxin/MADLAD_CultureX_cleaned/data/parquet')
OUTPUT_DIR = Path('/data/models/muxin/output')
LANGUAGE = "sin_Sinh"  # Sinhala language code

def prepare_laser3():
    """Initialize LASER3 environment"""
    print("Initializing LASER3...")
    start_time = time.time()
    
    os.environ['LASER'] = str(LASER_DIR)
    for path in [LASER_DIR / 'source', LASER_DIR / 'source/lib']:
        if str(path) not in sys.path:
            sys.path.append(str(path))
    
    from embed import SentenceEncoder
    
    vocab_path = NLLB_DIR / 'laser2.cvocab'
    print(f"Loading vocabulary from: {vocab_path}")
    
    encoder = SentenceEncoder(
        str(NLLB_DIR / 'laser3-sin_Sinh.v1.pt'),
        max_sentences=None,
        max_tokens=12000,
        cpu=False,
        spm_model=str(NLLB_DIR / 'laser2.spm'),
        spm_vocab=str(vocab_path)
    )
    
    elapsed = time.time() - start_time
    print(f"Initializing LASER3 completed after {elapsed:.2f}s.")
    return encoder


def process_madlad_data(output_dir, batch_size=50000, force_reprocess=False):
    """Process all .parquet files under MADLAD_DIR into smaller chunks with caching"""

    output_dir.mkdir(exist_ok=True, parents=True)
    cache_info_file = output_dir / 'cache_info.txt'

    # Gather all .parquet files under MADLAD_DIR (recursively)
    parquet_files = sorted([Path(p) for p in glob.glob(str(MADLAD_DIR / '**' / '*.parquet'), recursive=True)])
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found under {MADLAD_DIR}")

    # Compute a hash of all file paths and mtimes for cache validation
    hash_input = '|'.join(f"{str(f)}:{f.stat().st_mtime}" for f in parquet_files)
    files_hash = hashlib.md5(hash_input.encode()).hexdigest()[:10]

    # Check if cached data exists and is valid
    if not force_reprocess and cache_info_file.exists():
        with open(cache_info_file, 'r') as f:
            try:
                cache_info = f.read().strip()
                cached_hash, n_chunks = cache_info.split(',')
                n_chunks = int(n_chunks)
                if cached_hash == files_hash:
                    print(f"Using cached processed data with {n_chunks} chunks")
                    return n_chunks
            except Exception:
                print("Cache info file corrupted, reprocessing...")

    print("Processing MADLAD data from parquet files...")
    start_time = time.time()
    total_chunks = 0
    current_batch = []

    for parquet_file in tqdm(parquet_files, desc="Processing MADLAD files"):
        try:
            df = pd.read_parquet(parquet_file)
            # Try to find a column with sentences
            for col in ['text', 'sentence', 'sentences']:
                if col in df.columns:
                    sentences = df[col].astype(str).tolist()
                    break
            else:
                # If no known column, use the first column
                sentences = df.iloc[:, 0].astype(str).tolist()
        except Exception as e:
            print(f"Error reading {parquet_file}: {e}")
            continue

        for line in sentences:
            line = line.strip()
            if line:
                current_batch.append(line)
            if len(current_batch) >= batch_size:
                chunk_path = output_dir / f'chunk_{total_chunks:06d}.txt'
                with open(chunk_path, 'w', encoding='utf-8') as chunk_file:
                    chunk_file.write('\n'.join(current_batch))
                current_batch = []
                total_chunks += 1

    # Write remaining sentences
    if current_batch:
        chunk_path = output_dir / f'chunk_{total_chunks:06d}.txt'
        with open(chunk_path, 'w', encoding='utf-8') as chunk_file:
            chunk_file.write('\n'.join(current_batch))
        total_chunks += 1

    # Save cache info
    with open(cache_info_file, 'w') as f:
        f.write(f"{files_hash},{total_chunks}")

    elapsed = time.time() - start_time
    print(f"Processing MADLAD data completed after {elapsed:.2f}s.")
    return total_chunks

def process_cc100_data(output_dir, batch_size=50000, force_reprocess=False):
    """Process CC100 data into smaller chunks with caching"""
    output_dir.mkdir(exist_ok=True, parents=True)
    cache_info_file = output_dir / 'cache_info.txt'
    
    # Check if cached data exists and is valid
    if not force_reprocess and cache_info_file.exists():
        with open(cache_info_file, 'r') as f:
            try:
                cache_info = f.read().strip()
                source_mtime, n_chunks = cache_info.split(',')
                source_mtime = float(source_mtime)
                n_chunks = int(n_chunks)
                
                # Check if source file hasn't been modified
                if source_mtime == CC100_PATH.stat().st_mtime:
                    print(f"Using cached processed data with {n_chunks} chunks")
                    return n_chunks
            except:
                print("Cache info file corrupted, reprocessing...")
    
    print("Processing CC100 data...")
    start_time = time.time()
    total_chunks = 0
    
    with open(CC100_PATH, 'r', encoding='utf-8') as f:
        current_batch = []
        for line in tqdm(f, desc="Processing CC100"):
            line = line.strip()
            if line:  # Skip empty lines
                current_batch.append(line)
                
            if len(current_batch) >= batch_size:
                chunk_path = output_dir / f'chunk_{total_chunks:06d}.txt'
                with open(chunk_path, 'w', encoding='utf-8') as chunk_file:
                    chunk_file.write('\n'.join(current_batch))
                current_batch = []
                total_chunks += 1
        
        # Write remaining sentences
        if current_batch:
            chunk_path = output_dir / f'chunk_{total_chunks:06d}.txt'
            with open(chunk_path, 'w', encoding='utf-8') as chunk_file:
                chunk_file.write('\n'.join(current_batch))
            total_chunks += 1
    
    # Save cache info
    with open(cache_info_file, 'w') as f:
        f.write(f"{CC100_PATH.stat().st_mtime},{total_chunks}")
    
    elapsed = time.time() - start_time
    print(f"Processing CC100 data completed after {elapsed:.2f}s.")
    return total_chunks

def compute_embeddings(sentences, encoder, desc="Computing embeddings"):
    """Compute embeddings with caching from previous runs"""
    if isinstance(sentences, (str, Path)):
        with open(sentences, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
    
    sentences = [s.strip() for s in sentences]
    total_sentences = len(sentences)
    
    # Create a hash for these sentences
    if isinstance(sentences, list):
        # Use first 1000 sentences and length as hash input for large lists
        hash_input = '\n'.join(sentences[:1000]) + f"|{len(sentences)}"
        sentences_hash = hashlib.md5(hash_input.encode()).hexdigest()[:10]
    else:
        sentences_hash = get_file_hash(sentences)
    
    # Check if we have a complete cache for these sentences
    embeddings_cache_dir = OUTPUT_DIR / 'embeddings_cache'
    embeddings_cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file = embeddings_cache_dir / f'embed_{sentences_hash}.npy'
    
    if cache_file.exists():
        print(f"Loading cached embeddings from {cache_file}")
        return np.load(cache_file)
    
    # If no cached file, continue with checkpointing as before
    checkpoint_dir = OUTPUT_DIR / 'checkpoints' / 'embeddings'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_file = checkpoint_dir / f'progress_{sentences_hash}.json'
    
    # Load checkpoint if exists
    batch_size = 10000
    n_batches = math.ceil(total_sentences / batch_size)
    all_embeddings = [None] * n_batches  # Pre-allocate list
    completed_batches = set()
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                completed_batches = set(checkpoint_data.get('completed_batches', []))
                print(f"Loaded checkpoint with {len(completed_batches)} completed batches")
                
                # Load completed batch embeddings
                for batch_id in completed_batches:
                    batch_file = checkpoint_dir / f'batch_{sentences_hash}_{batch_id}.npy'
                    if batch_file.exists():  # Add existence check
                        all_embeddings[int(batch_id)] = np.load(batch_file)
                    else:
                        completed_batches.remove(batch_id)  # Remove if file missing
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            completed_batches = set()
    
    print(f"\nComputing embeddings for {total_sentences} sentences in {n_batches} batches...")
    
    with tqdm(total=total_sentences, desc=desc) as pbar:
        # Update progress bar for already completed batches
        initial_progress = sum(min((i+1)*batch_size, total_sentences) - i*batch_size 
                            for i in range(n_batches) if str(i) in completed_batches)
        pbar.update(initial_progress)
        
        for i in range(n_batches):
            if str(i) in completed_batches:
                continue
                
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_sentences)
            batch = sentences[start_idx:end_idx]
            
            print(f"\nProcessing batch {i+1}/{n_batches} ({len(batch)} sentences)")
            
            try:
                batch_embeddings = encoder.encode_sentences(batch)
                all_embeddings[i] = batch_embeddings
                
                # Save batch embeddings
                batch_file = checkpoint_dir / f'batch_{sentences_hash}_{i}.npy'
                np.save(batch_file, batch_embeddings)
                
                # Update checkpoint
                completed_batches.add(str(i))
                with open(checkpoint_file, 'w') as f:
                    json.dump({'completed_batches': list(completed_batches)}, f)
                
                pbar.update(len(batch))
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                print(f"Continuing with next batch...")
                continue
    
    # Combine all embeddings and verify shapes
    valid_embeddings = [emb for emb in all_embeddings if emb is not None]
    if not valid_embeddings:
        raise RuntimeError("No valid embeddings were computed!")
    
    final_embeddings = np.vstack(valid_embeddings)
    print(f"Completed embedding computation. Shape: {final_embeddings.shape}")
    
    # Save to cache for future use
    print(f"Saving embeddings to cache: {cache_file}")
    np.save(cache_file, final_embeddings)
    
    # Clean up checkpoints only if all batches completed successfully
    if len(completed_batches) == n_batches:
        try:
            for batch_file in checkpoint_dir.glob(f'batch_{sentences_hash}_*.npy'):
                batch_file.unlink()
            checkpoint_file.unlink()
            if not any(checkpoint_dir.iterdir()):
                checkpoint_dir.rmdir()
        except Exception as e:
            print(f"Error cleaning up checkpoint files: {e}")
    
    return final_embeddings


def main():
    # Setup directories
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    processed_data_dir = OUTPUT_DIR / 'processed_data'
    dataset_dir = get_dataset_dir('uts') / 'si'  # Sinhala dataset directory
    
    # Initialize LASER3
    encoder = prepare_laser3()
    
    # # Process CC100 data with caching
    # n_chunks = process_cc100_data(processed_data_dir)
    # print(f"Working with {n_chunks} chunks of data")
    
    # Process MADLAD data
    n_chunks = process_madlad_data(processed_data_dir)
    print(f"Working with {n_chunks} chunks of data")
    
    # Create base index
    with log_action('Creating base index'):
        n_train_sentences = 10**7
        train_sentences = []
        chunk_paths = list(processed_data_dir.glob('chunk_*.txt'))
        
        print("\nCollecting training sentences...")
        with tqdm(total=n_train_sentences, desc="Loading sentences") as pbar:
            for chunk_path in chunk_paths:
                for sentence in yield_lines(chunk_path):
                    train_sentences.append(sentence)
                    pbar.update(1)
                    if len(train_sentences) == n_train_sentences:
                        break
                if len(train_sentences) == n_train_sentences:
                    break
        
        base_index_dir = dataset_dir / 'base_indexes'
        base_index_dir.mkdir(exist_ok=True, parents=True)
        
        # Create base index using LASER3 embeddings
        base_index_path = create_base_index(
            train_sentences,
            get_index_name(),
            lambda s: compute_embeddings(s, encoder),
            faiss.METRIC_L2,
            base_index_dir
        )
    
    # Compute embeddings for all chunks
    with log_action('Computing embeddings'):
        cache_dir = get_cache_dir(dataset_dir) / 'laser_si'
        indexes_dir = cache_dir / 'indexes' / f'base-index-{get_file_hash(base_index_path)}'
        indexes_dir.mkdir(exist_ok=True, parents=True)
        
        # Process each chunk
        for chunk_path in tqdm(chunk_paths, desc="Processing chunks"):
            if get_index_path(chunk_path, indexes_dir).exists():
                continue
            
            # Compute and save embeddings for the chunk
            compute_and_save_embeddings(
                chunk_path,
                base_index_path,
                lambda s: compute_embeddings(s, encoder),
                indexes_dir=indexes_dir
            )
    
    # Mine paraphrases
    with log_action('Mining paraphrases'):
        nn_search_results_dir = cache_dir / 'nn_search_results'
        nn_search_results_dir.mkdir(exist_ok=True, parents=True)
        
        topk = 8
        nprobe = 16
        
        # Run NN search for each chunk
        for query_chunk in tqdm(chunk_paths, desc="Mining paraphrases"):
            if get_results_path(query_chunk, chunk_paths, topk, nprobe, nn_search_results_dir).exists():
                continue
            
            compute_and_save_nn_batched(
                query_chunk,
                chunk_paths,
                topk,
                nprobe,
                indexes_dir,
                nn_search_results_dir,
                delete_intermediary=True
            )
            # compute_and_save_nn_batched_parallel(
            # query_chunk,
            # chunk_paths,
            # topk,
            # nprobe,
            # indexes_dir,
            # nn_search_results_dir,
            # n_jobs=24,
            # delete_intermediary=True
            # )
            
    # Replace sequential processing with parallel jobs
    # with log_action('Mining paraphrases'):
    #     nn_search_results_dir = cache_dir / 'nn_search_results'
    #     nn_search_results_dir.mkdir(exist_ok=True, parents=True)
        
    #     topk = 8
    #     nprobe = 16
        
    #     # Create a list of jobs to run
    #     jobs_to_run = []
    #     for query_chunk in chunk_paths:
    #         if not get_results_path(query_chunk, chunk_paths, topk, nprobe, nn_search_results_dir).exists():
    #             jobs_to_run.append((query_chunk, chunk_paths, topk, nprobe, indexes_dir, nn_search_results_dir))
        
    #     # Create executor with appropriate settings for your environment
    #     executor = get_executor(
    #         folder=str(OUTPUT_DIR / 'logs'),
    #         cluster='local',  # Use local execution
    #         gpus_per_node=1,  # Adjust this to match your available GPUs
    #         timeout_min=720,  # Adjust timeout as needed
    #         cpus_per_task=10, # Adjust CPU count as needed
    #         slurm_max_num_timeout=3,  # Number of times to retry on timeout
    #         max_local_workers=4,
    #     )
        
    #     # Submit jobs in parallel
    #     if jobs_to_run:
    #         print(f"Submitting {len(jobs_to_run)} parallel jobs for NN search...")
    #         jobs = []
    #         for job_args in jobs_to_run:
    #             job = executor.submit(compute_and_save_nn_batched, *job_args, delete_intermediary=True)
    #             jobs.append(job)
            
    #         # Wait for all jobs to complete
    #         for i, job in enumerate(jobs):
    #             print(f"Waiting for job {i+1}/{len(jobs)}...")
    #             job.result()
    
    # Filter candidate paraphrases
    with log_action('Filtering candidate paraphrases'):
        pairs_dir = cache_dir / 'pairs'
        pairs_dir.mkdir(exist_ok=True, parents=True)
        
        filter_kwargs = {
            'density': 0.6,
            'distance': 0.05,
            'levenshtein': 0.2,
            'simplicity': 0.0,
            'filter_ne': False,
        }
        
        # Process each chunk for paraphrase pairs
        for query_chunk in tqdm(chunk_paths, desc="Filtering paraphrases"):
            simplification_pairs_path = get_pairs_path(
                query_chunk,
                chunk_paths,
                topk,
                nprobe,
                filter_kwargs,
                pairs_dir
            )
            
            if simplification_pairs_path.exists():
                continue
            
            compute_and_save_simplification_pairs(
                query_chunk,
                chunk_paths,
                base_index_path,
                cache_dir,
                pairs_dir,
                lambda s: compute_embeddings(s, encoder),
                topk,
                nprobe,
                'si',
                filter_kwargs,
                lambda pair: True  # No simplicity filtering for paraphrases
            )
            
    # with log_action('Filtering candidate paraphrases'):
    #     pairs_dir = cache_dir / 'pairs'
    #     pairs_dir.mkdir(exist_ok=True, parents=True)
        
    #     filter_kwargs = {
    #         'density': 0.6,
    #         'distance': 0.05,
    #         'levenshtein': 0.2,
    #         'simplicity': 0.0,
    #         'filter_ne': False,
    #     }
        
    #     # For paraphrases, we don't need simplicity filtering
    #     is_simpler = lambda pair: True  # No simplicity filtering for paraphrases
        
    #     # Check which chunks need filtering
    #     chunks_to_process = []
    #     for query_chunk in tqdm(chunk_paths, desc="Checking chunks for filtering"):
    #         simplification_pairs_path = get_pairs_path(
    #             query_chunk,
    #             chunk_paths,
    #             topk,
    #             nprobe,
    #             filter_kwargs,
    #             pairs_dir
    #         )
            
    #         if not simplification_pairs_path.exists():
    #             chunks_to_process.append(query_chunk)
        
    #     print(f"Found {len(chunks_to_process)} chunks to process for filtering")
        
    #     if chunks_to_process:
    #         # Define filtering function
    #         def process_chunk_filtering(query_chunk):
    #             try:
    #                 return compute_and_save_simplification_pairs(
    #                     query_chunk,                                 # query_sentences_path
    #                     chunk_paths,                                # db_sentences_paths
    #                     base_index_path,                            # base_index_path
    #                     lambda s: compute_embeddings(s, encoder),   # get_embeddings
    #                     cache_dir,                                  # cache_dir
    #                     pairs_dir,                                  # pairs_dir
    #                     topk,                                       # topk
    #                     nprobe,                                     # nprobe
    #                     'si',                                       # language code for Sinhala
    #                     filter_kwargs,                              # filter_kwargs
    #                     is_simpler,                                 # is_simpler
    #                 )
    #             except Exception as e:
    #                 print(f"Error processing chunk {query_chunk}: {e}")
    #                 raise
            
    #         # Process filtering in parallel
    #         from concurrent.futures import ThreadPoolExecutor
    #         import time
            
    #         start_time = time.time()
    #         max_workers = min(8, len(chunks_to_process))  # Use up to 8 workers for filtering
            
    #         print(f"Processing filtering with {max_workers} parallel workers...")
            
    #         with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #             futures = [
    #                 executor.submit(process_chunk_filtering, chunk)
    #                 for chunk in chunks_to_process
    #             ]
                
    #             # Monitor progress with detailed timing
    #             completed = 0
    #             chunk_times = []
                
    #             with tqdm(total=len(futures), desc="Filtering paraphrases") as pbar:
    #                 for future in concurrent.futures.as_completed(futures):
    #                     try:
    #                         result = future.result()
    #                         completed += 1
                            
    #                         # Calculate ETA for filtering
    #                         elapsed = time.time() - start_time
    #                         if completed > 0:
    #                             avg_time_per_chunk = elapsed / completed
    #                             remaining = len(chunks_to_process) - completed
    #                             eta_seconds = avg_time_per_chunk * remaining
    #                             eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                                
    #                             pbar.set_description(
    #                                 f"Filtering paraphrases - {completed}/{len(chunks_to_process)} - ETA: {eta}"
    #                             )
                            
    #                         pbar.update(1)
                            
    #                     except Exception as e:
    #                         print(f"Filtering chunk failed: {e}")
    #                         raise
            
    #         filtering_time = time.time() - start_time
    #         print(f"Completed filtering for {len(chunks_to_process)} chunks in {filtering_time:.2f}s")
    
    
    # Combine results
    with log_action('Wrapping up paraphrases'):
        simplification_pairs = get_simplification_pairs_paths(
            chunk_paths,
            chunk_paths,
            topk,
            nprobe,
            filter_kwargs,
            pairs_dir
        )
        
        results_str = f'query-{get_files_hash(chunk_paths)}_db-{get_files_hash(chunk_paths)}_topk-{topk}_nprobe-{nprobe}'
        filter_str = get_filter_string_representation(filter_kwargs)
        
        final_path = dataset_dir / f'paraphrases_{results_str}_{filter_str}.jsonl'
        combine_simplifications_in_dataset(simplification_pairs, final_path)
        print(f"\nFinal dataset saved to: {final_path}")

if __name__ == '__main__':
    main()