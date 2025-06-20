# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from faiss.loader import *

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm
import torch
from joblib import Parallel, delayed

from muss.utils.helpers import (
    get_files_hash,
    log_action,
    write_lines,
    yield_lines,
    print_running_time,
    count_lines,
    get_file_hash,
    read_lines,
    lock_file,
    create_directory_or_skip,
    write_lines_in_parallel,
)
from muss.resources.paths import get_dataset_dir, get_data_filepath
from muss.mining.filtering import (
    is_contained,
    is_overlapping,
    is_different_enough,
    filter_candidate_pairs,
    has_hallucinated_named_entities,
)


def get_cache_dir(dataset_dir):
    return dataset_dir / 'cache'


@lru_cache(maxsize=10000)
def cached_count_lines(filepath):
    '''We store the result because its quite long to compute compared to how often we need it'''
    line_count_path = filepath.parent / f'.file-{get_file_hash(filepath)}.line_counts.txt'
    line_count_path.parent.mkdir(exist_ok=True, parents=True)
    if not line_count_path.exists():
        line_count_path.touch()
        with lock_file(line_count_path):
            line_count = count_lines(filepath)
            with open(line_count_path, 'w') as f:
                f.write(f'{line_count}\n')
    # TODO: Ideally we should lock the file when reading as well but it's slow
    with open(line_count_path, 'r') as f:
        try:
            return int(f.read().strip('\n'))
        except ValueError:  # Sometimes the file is empty
            line_count_path.unlink()
            return cached_count_lines(filepath)


def load_index(index_path):
    co = faiss.GpuMultipleClonerOptions()
    co.use_cuvs = False

    idx = faiss.read_index(str(index_path))
    
    return faiss.index_cpu_to_all_gpus(idx, co=co)


def load_indexes(index_paths):
    index = load_index(index_paths[0])
    pbar = tqdm(index_paths[1:])
    for index_path in pbar:
        pbar.set_description_str('Loading from ' + os.path.basename(index_path))
        faiss.merge_into(index, load_index(index_path), True)

    return index
    # return faiss.index_cpu_to_all_gpus(index)
    
# def load_indexes(index_paths, gpu_id=0, gpu_resources=None, config=None):
#     """Load and merge multiple indexes with explicit GPU control"""
#     # Load first index
#     index = load_index(index_paths[0])
    
#     # Merge additional indexes
#     pbar = tqdm(index_paths[1:])
#     for index_path in pbar:
#         pbar.set_description_str('Loading from ' + os.path.basename(index_path))
#         faiss.merge_into(index, load_index(index_path), True)
    
#     # Convert to GPU index with explicit control
#     if gpu_resources is None:
#         # Fallback to default behavior
#         return faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), gpu_id, index)
#     else:
#         # Use provided resources and config
#         if config is None:
#             return faiss.index_cpu_to_gpu(gpu_resources, gpu_id, index)
#         else:
#             return faiss.index_cpu_to_gpu(gpu_resources, gpu_id, index, config)

def get_index_path(sentences_path, indexes_dir):
    return Path(indexes_dir) / f'sentences-{get_file_hash(sentences_path)}.faiss_index'


def get_results_string_representation(query_sentences_path, db_sentences_paths, topk, nprobe):
    query_hash = get_file_hash(query_sentences_path)
    db_hash = get_files_hash(db_sentences_paths)
    return f'query-{query_hash}_db-{db_hash}_topk-{topk}_nprobe-{nprobe}'


def get_results_path(query_sentences_path, db_sentences_paths, topk, nprobe, nn_search_results_dir):
    return (
        nn_search_results_dir
        / f'{get_results_string_representation(query_sentences_path, db_sentences_paths, topk, nprobe)}.npz'
    )


def compute_and_save_embeddings(sentences_path, base_index_path, get_embeddings, indexes_dir):
    index_path = get_index_path(sentences_path, indexes_dir)
    if not index_path.exists():
        with log_action('Computing and saving embeddings'):
            sentences = read_lines(sentences_path)
            embeddings = get_embeddings(sentences)
            index = load_index(base_index_path)
            index.add(embeddings)
            index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index, str(index_path))
    return index_path


def get_nearest_sentence_ids(query_index, db_index, topk, nprobe, batch_size=1024, use_gpu=True):
    try:
        faiss.ParameterSpace().set_index_parameter(db_index, 'nprobe', nprobe)
    except RuntimeError as e:
        if 'could not set parameter nprobe' in str(e):
            pass
        else:
            raise e
    # if use_gpu:
        # db_index = faiss.index_cpu_to_all_gpus(db_index)
    all_distances = np.empty((query_index.ntotal, topk))
    all_sentence_ids = np.empty((query_index.ntotal, topk), dtype=int)
    for batch_idx in tqdm(range((query_index.ntotal // batch_size) + 1), desc='Compute nearest neighbors'):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, query_index.ntotal)
        actual_batch_size = end_idx - start_idx
        query_embeddings = query_index.reconstruct_n(start_idx, actual_batch_size)  # TODO: Do this in the background
        distances, sentence_ids = db_index.search(query_embeddings, topk)
        all_distances[start_idx:end_idx] = distances
        all_sentence_ids[start_idx:end_idx] = sentence_ids
    # If distances are sorted in descending order, we make them ascending instead for the following code to work
    if np.all(np.diff(all_distances) <= 0):
        # This is taylored for transforming cosine similarity into a pseudo-distance: the maximum cosine similarity is 1 (vectors are equal).
        # Hence distance = 1 - cosine will always be positive and will be be equal to 0 when vectors are equal.
        all_distances = 1 - all_distances
    return all_distances, all_sentence_ids.astype(int)


def dump_results(distances, sentence_ids, results_path):
    np.savez(results_path, distances=distances, sentence_ids=sentence_ids, allow_pickle=False)
    return results_path


def load_results(results_path):
    assert results_path.exists(), f'Results path does not exist.\nresults_path={results_path}'
    try:
        results = np.load(results_path, allow_pickle=False)
        return results['distances'], results['sentence_ids']
    except Exception:
        print(f'results_path={results_path}')
        results_path.unlink()
        raise


def compute_and_save_nn(query_sentences_path, db_sentences_paths, topk, nprobe, indexes_dir, nn_search_results_dir):
    results_path = get_results_path(query_sentences_path, db_sentences_paths, topk, nprobe, nn_search_results_dir)
    if results_path.exists():
        return results_path
    
    query_index = load_index(get_index_path(query_sentences_path, indexes_dir))
    db_index = load_indexes([get_index_path(sentences_path, indexes_dir) for sentences_path in db_sentences_paths])
    distances, sentence_ids = get_nearest_sentence_ids(query_index, db_index, topk, nprobe)
    
    dump_results(distances, sentence_ids, results_path)
    
    return results_path


def combine_results_over_db_indexes(intermediary_results_paths, offsets):
    def combine_distances_and_ids(distances_list, sentence_ids_list):
        topk = distances_list[0].shape[1]
        assert all(distances.shape[1] == topk for distances in distances_list)
        distances = np.concatenate(distances_list, axis=1)
        sentence_ids = np.concatenate(sentence_ids_list, axis=1)
        kept_indexes = np.argsort(distances, axis=1)[:, :topk]
        return np.take_along_axis(distances, kept_indexes, axis=1), np.take_along_axis(
            sentence_ids, kept_indexes, axis=1
        )

    for i, (results_path, offset) in tqdm(
        list(enumerate(zip(intermediary_results_paths, offsets))), desc='Combine db indexes'
    ):
        distances, sentence_ids = load_results(results_path)
        sentence_ids += offset
        if i == 0:
            # No need to combine at first iteration
            all_distances, all_sentence_ids = distances, sentence_ids
            continue
        all_distances, all_sentence_ids = combine_distances_and_ids(
            [all_distances, distances], [all_sentence_ids, sentence_ids]
        )
    return all_distances, all_sentence_ids


def compute_and_save_nn_batched(
    query_sentences_path,
    db_sentences_paths,
    topk,
    nprobe,
    indexes_dir,
    nn_search_results_dir,
    n_samples_per_gpu=1e7,
    delete_intermediary=True,
):
    combined_results_path = get_results_path(
        query_sentences_path, db_sentences_paths, topk, nprobe, nn_search_results_dir
    )
    if combined_results_path.exists():
        return combined_results_path
    # Batch db paths to fit on one GPU
    db_sentences_paths_batches = []
    batch = []
    n_batch_samples = 0
    for db_sentences_path in tqdm(db_sentences_paths, desc='Batching db files'):
        n_samples = cached_count_lines(db_sentences_path)
        if n_batch_samples + n_samples > n_samples_per_gpu:
            db_sentences_paths_batches.append(batch)
            batch = []
            n_batch_samples = 0
        batch.append(db_sentences_path)
        n_batch_samples += n_samples
    db_sentences_paths_batches.append(batch)
    intermediary_results_paths = []
    offset = 0
    offsets = []
    for db_sentences_paths_batch in tqdm(db_sentences_paths_batches, desc='Compute NN db batches'):        
        intermediary_results_path = compute_and_save_nn(
            query_sentences_path, db_sentences_paths_batch, topk, nprobe, indexes_dir, nn_search_results_dir
        )
        intermediary_results_paths.append(intermediary_results_path)
        offsets.append(offset)
        offset += sum([cached_count_lines(sentences_path) for sentences_path in db_sentences_paths_batch])
    if len(intermediary_results_paths) == 1:
        assert combined_results_path == intermediary_results_paths[0]
    else:
        # Combine and save final result
        distances, sentence_ids = print_running_time(combine_results_over_db_indexes)(
            intermediary_results_paths, offsets
        )
        dump_results(distances, sentence_ids, combined_results_path)
        # Delete intermediary results
        if delete_intermediary:
            for intermediary_results_path in intermediary_results_paths:
                intermediary_results_path.unlink()
    return combined_results_path


def get_candidate_pair_ids(distances, sentence_ids, distance_threshold, density_threshold):
    distances = torch.FloatTensor(distances)
    sentence_ids = torch.IntTensor(sentence_ids)
    mean_distances = distances.mean(axis=1)
    densities = distances / mean_distances.reshape(-1, 1)
    # Take only ids that are close enough
    query_sentence_ids = torch.arange(sentence_ids.shape[0]).reshape(-1, 1).repeat(1, sentence_ids.shape[1])
    lower_distance_threshold = 1e-10  # To filter out exact matches
    mask = (
        (np.abs(distances) > lower_distance_threshold)
        & (distances < distance_threshold)
        & (densities < density_threshold)
    )
    nearest_sentence_ids = torch.masked_select(sentence_ids, mask)
    query_sentence_ids = torch.masked_select(query_sentence_ids, mask)
    # Remove queries that were aligned with a sentence from the same document (we make sure their sentence ids are far enough appart). This prevents overlaps
    mask = (query_sentence_ids - nearest_sentence_ids).abs() > 10
    query_sentence_ids = query_sentence_ids[mask]
    nearest_sentence_ids = nearest_sentence_ids[mask]
    return query_sentence_ids.numpy().astype(int), nearest_sentence_ids.numpy().astype(int)


def get_sentences_from_ids(sentence_ids, sentences_paths):
    def get_sentences_from_ids_single_file(sentence_ids, sentences_path):
        sentences = read_lines(sentences_path)
        try:
            return [sentences[sentence_id] for sentence_id in sentence_ids]
        except IndexError:
            print(
                f'len(sentences)={len(sentences)}, max(sentence_ids)={max(sentence_ids)}, sentences_path={sentences_path}'
            )
            raise

    sorted_idx = np.argsort(sentence_ids)
    sentence_ids = sentence_ids[sorted_idx]
    ids_per_file = defaultdict(list)
    n_sentences_list = [cached_count_lines(sentences_path) for sentences_path in sentences_paths]
    offsets = np.cumsum([0] + n_sentences_list[:-1])
    next_offsets = np.cumsum(n_sentences_list)
    sentence_ids = np.sort(sentence_ids)
    for offset, next_offset, sentences_path in zip(offsets, next_offsets, sentences_paths):
        selected_sentence_ids = sentence_ids[(offset <= sentence_ids) & (sentence_ids < next_offset)]
        if len(selected_sentence_ids) > 0:
            selected_sentence_ids -= offset
            ids_per_file[sentences_path].extend(selected_sentence_ids.tolist())
    # The sentences should be returned in the correct order because python dicts are insertion ordered
    sentences_list = Parallel(n_jobs=10)(
        delayed(get_sentences_from_ids_single_file)(sentence_ids, sentences_path)
        for sentences_path, sentence_ids in tqdm(ids_per_file.items(), desc='Load sentences')
    )
    sentences = [sentence for sentences in sentences_list for sentence in sentences]
    # Put sentences back in order
    return [sentences[idx] for idx in np.argsort(sorted_idx)]


def combine_results_over_queries(query_sentences_paths, db_sentences_paths, topk, nprobe, nn_search_results_dir):
    all_distances = np.empty((0, topk), dtype=float)
    all_sentence_ids = np.empty((0, topk), dtype=int)
    for query_sentences_path in tqdm(query_sentences_paths, desc='Combine queries'):
        results_path = get_results_path(query_sentences_path, db_sentences_paths, topk, nprobe, nn_search_results_dir)
        distances, sentence_ids = load_results(results_path)
        all_distances = np.concatenate([all_distances, distances], axis=0)
        all_sentence_ids = np.concatenate([all_sentence_ids, sentence_ids], axis=0)
    return all_distances, all_sentence_ids


def find_nearest_neighbors(
    query_sentences_paths,
    db_sentences_paths,
    base_index_path,
    get_embeddings,
    cache_dir,
    topk=8,
    nprobe=16,
    distance_threshold=0.1,
    density_threshold=0.8,
):
    query_sentences_paths = list(sorted(query_sentences_paths))
    db_sentences_paths = list(sorted(db_sentences_paths))
    # Create indexes
    indexes_dir = cache_dir / 'indexes' / f'base-index-{get_file_hash(base_index_path)}'
    indexes_dir.mkdir(exist_ok=True, parents=True)
    for sentences_path in query_sentences_paths + db_sentences_paths:
        compute_and_save_embeddings(sentences_path, base_index_path, get_embeddings, indexes_dir=indexes_dir)
    # Run NN search query file by query file
    nn_search_results_dir = cache_dir / 'nn_search_results'
    nn_search_results_dir.mkdir(exist_ok=True, parents=True)
    for query_sentences_path in query_sentences_paths:
        compute_and_save_nn_batched(
            query_sentences_path, db_sentences_paths, topk, nprobe, indexes_dir, nn_search_results_dir
        )
    # Retrieve sentences
    distances, sentence_ids = combine_results_over_queries(
        query_sentences_paths, db_sentences_paths, topk, nprobe, nn_search_results_dir
    )
    query_sentence_ids, db_sentence_ids = get_candidate_pair_ids(
        distances, sentence_ids, distance_threshold=distance_threshold, density_threshold=density_threshold
    )
    query_sentences = get_sentences_from_ids(query_sentence_ids, query_sentences_paths)
    db_sentences = get_sentences_from_ids(db_sentence_ids, db_sentences_paths)
    return list(zip(query_sentences, db_sentences))


def get_pairs_path(query_sentences_path, db_sentences_paths, topk, nprobe, filter_kwargs, pairs_dir):
    results_str = get_results_string_representation(query_sentences_path, db_sentences_paths, topk, nprobe)
    filter_str = get_filter_string_representation(filter_kwargs)
    return pairs_dir / f'pairs_{results_str}_{filter_str}.tsv'


def get_paraphrase_pairs(
    query_sentences_path, db_sentences_paths, base_index_path, get_embeddings, cache_dir, topk, nprobe, filter_kwargs
):
    candidate_pairs = print_running_time(find_nearest_neighbors)(
        [query_sentences_path],
        db_sentences_paths,
        base_index_path,
        get_embeddings,
        cache_dir,
        topk=topk,
        nprobe=nprobe,
        distance_threshold=filter_kwargs['distance'],
        density_threshold=filter_kwargs['density'],
    )
    print(f'#candidates: {len(candidate_pairs)}')
    filters = {
        'macro-duplicates': lambda pairs: list(set(candidate_pairs)),
        'is_contained': lambda pair: not is_contained(*pair),
        'is_overlapping': lambda pair: not is_overlapping(*pair),
    }
    if filter_kwargs.get('levenshtein', 0) > 0:
        filters['is_different_enough'] = lambda pair: is_different_enough(*pair, threshold=filter_kwargs['levenshtein'])
    with log_action('filtering'):
        return filter_candidate_pairs(candidate_pairs, filters)


def compute_and_save_simplification_pairs(
    query_sentences_path,
    db_sentences_paths,
    base_index_path,
    get_embeddings,
    cache_dir,
    pairs_dir,
    topk,
    nprobe,
    language,
    filter_kwargs,
    is_simpler,
):
    simplifications_path = get_pairs_path(
        query_sentences_path, db_sentences_paths, topk, nprobe, filter_kwargs, pairs_dir
    )
    if not simplifications_path.exists():
        paraphrase_pairs = get_paraphrase_pairs(
            query_sentences_path,
            db_sentences_paths,
            base_index_path,
            get_embeddings,
            cache_dir,
            topk,
            nprobe,
            filter_kwargs,
        )
        filters = {
            'is_simpler': is_simpler,
        }
        if filter_kwargs.get('filter_ne', True):
            filters['has_hallucinated_ne'] = lambda pair: not has_hallucinated_named_entities(*pair, language=language)
        with log_action('filtering'):
            simplification_pairs = filter_candidate_pairs(paraphrase_pairs, filters)
        write_pairs_to_file(simplification_pairs, simplifications_path)
    return simplifications_path


def write_pairs_to_file(pairs, filepath):
    write_lines(('\t'.join(pair) for pair in pairs), filepath)


def yield_pairs_from_file(filepath):
    for line in yield_lines(filepath):
        complex_sentence, simple_sentence = line.split('\t')
        yield (complex_sentence, simple_sentence)


def get_pairs_from_file(filepath):
    return list(yield_pairs_from_file(filepath))


def get_filter_string_representation(filter_kwargs):
    return '_'.join(sorted([f'{key}-{value}' for key, value in filter_kwargs.items()]))


def get_simplification_pairs_paths(query_sentences_paths, db_sentences_paths, topk, nprobe, filter_kwargs, pairs_dir):
    simplification_pairs = []
    for query_sentences_path in tqdm(query_sentences_paths, desc='Read queries'):
        simplification_pairs_path = get_pairs_path(
            query_sentences_path, db_sentences_paths, topk, nprobe, filter_kwargs, pairs_dir
        )
        simplification_pairs.extend(get_pairs_from_file(simplification_pairs_path))
    return simplification_pairs


def combine_simplifications_in_dataset(simplification_pairs, dataset):
    with create_directory_or_skip(get_dataset_dir(dataset)):
        assert len(simplification_pairs) > 30000, f'Not enough pairs: {len(simplification_pairs)}'
        indexes = np.random.permutation(len(simplification_pairs))
        for phase, start_index, end_index in [
            ('test', 10000, 20000),
            ('valid', 20000, 30000),
            ('train', 30000, len(indexes)),
        ]:
            with write_lines_in_parallel(
                [get_data_filepath(dataset, phase, 'complex'), get_data_filepath(dataset, phase, 'simple')]
            ) as files:
                for idx in tqdm(indexes[start_index:end_index]):
                    files.write(simplification_pairs[idx])
    return get_dataset_dir(dataset)


# def compute_and_save_nn_batched_parallel(
#     query_sentences_path,
#     db_sentences_paths,
#     topk,
#     nprobe,
#     indexes_dir,
#     nn_search_results_dir,
#     n_samples_per_gpu=1e5,
#     delete_intermediary=True,
#     n_jobs=3,
# ):
#     combined_results_path = get_results_path(
#         query_sentences_path, db_sentences_paths, topk, nprobe, nn_search_results_dir
#     )
#     if combined_results_path.exists():
#         return combined_results_path
        
#     # Batch db paths to fit on one GPU
#     db_sentences_paths_batches = []
#     batch = []
#     n_batch_samples = 0
#     for db_sentences_path in tqdm(db_sentences_paths, desc='Batching db files'):
#         n_samples = cached_count_lines(db_sentences_path)
#         if n_batch_samples + n_samples > n_samples_per_gpu:
#             db_sentences_paths_batches.append(batch)
#             batch = []
#             n_batch_samples = 0
#         batch.append(db_sentences_path)
#         n_batch_samples += n_samples
#     db_sentences_paths_batches.append(batch)
    
#     # Set up data for parallel jobs
#     intermediary_results_paths = []
#     offsets = []
#     offset = 0
#     for db_batch in db_sentences_paths_batches:
#         # Generate expected result path for each batch
#         result_path = get_results_path(
#             query_sentences_path, db_batch, topk, nprobe, nn_search_results_dir
#         )
#         intermediary_results_paths.append(result_path)
#         offsets.append(offset)
#         offset += sum([cached_count_lines(sentences_path) for sentences_path in db_batch])
    
#     # Process batches that don't already exist
#     batches_to_process = []
#     for i, (batch, result_path) in enumerate(zip(db_sentences_paths_batches, intermediary_results_paths)):
#         if not result_path.exists():
#             batches_to_process.append((i, batch, result_path))
    
#     if batches_to_process:
#         print(f"Processing {len(batches_to_process)} batches in parallel")
        
#         # Define the function to process one batch
#         def process_batch(batch_idx, db_batch, result_path):
#             # Make sure to use a specific GPU
#             # gpu_id = batch_idx % n_jobs
#             gpu_id = batch_idx % torch.cuda.device_count()
#             print(f"Processing batch {batch_idx}")
#             print(f"CUDA device count: {torch.cuda.device_count()}")
#             print(f"Attempting to use GPU {gpu_id}")
#             print(f"Current device: {torch.cuda.current_device()}")
#             with torch.cuda.device(gpu_id):
#                 print(f"Processing batch {batch_idx} on GPU {gpu_id}")
                
#                 # Load and prepare indexes
#                 query_index = load_index(get_index_path(query_sentences_path, indexes_dir))
#                 db_paths = [get_index_path(path, indexes_dir) for path in db_batch]
#                 db_index = load_indexes(db_paths, gpu_id)
                
#                 # Get nearest neighbors
#                 distances, sentence_ids = get_nearest_sentence_ids(
#                     query_index, db_index, topk, nprobe
#                 )
                
#                 # Save results
#                 dump_results(distances, sentence_ids, result_path)
                
#                 return result_path
        
#         # Run in parallel with joblib
#         Parallel(n_jobs=n_jobs, backend="threading")(
#             delayed(process_batch)(batch_idx, db_batch, result_path) 
#             for batch_idx, db_batch, result_path in batches_to_process
#         )
    
#     # Check if all results were generated
#     for path in intermediary_results_paths:
#         if not path.exists():
#             raise RuntimeError(f"Expected result file wasn't created: {path}")
    
#     if len(intermediary_results_paths) == 1:
#         # If only one batch, no need to combine
#         if combined_results_path != intermediary_results_paths[0]:
#             import shutil
#             shutil.copy(intermediary_results_paths[0], combined_results_path)
#     else:
#         # Combine and save final result
#         distances, sentence_ids = print_running_time(combine_results_over_db_indexes)(
#             intermediary_results_paths, offsets
#         )
#         dump_results(distances, sentence_ids, combined_results_path)
        
#         # Delete intermediary results
#         if delete_intermediary:
#             for intermediary_results_path in intermediary_results_paths:
#                 try:
#                     intermediary_results_path.unlink()
#                 except:
#                     pass
                    
#     return combined_results_path

def compute_and_save_nn_batched_parallel(
    query_sentences_path,
    db_sentences_paths,
    topk,
    nprobe,
    indexes_dir,
    nn_search_results_dir,
    n_samples_per_gpu=5e6,
    delete_intermediary=True,
    n_jobs=24,
):
    # Import necessary libraries
    from tqdm import tqdm
    import time
    import datetime
    import concurrent.futures
    
    # Track overall start time
    overall_start_time = time.time()
    
    combined_results_path = get_results_path(
        query_sentences_path, db_sentences_paths, topk, nprobe, nn_search_results_dir
    )
    if combined_results_path.exists():
        return combined_results_path
        
    # Batch db paths to fit on one GPU
    db_sentences_paths_batches = []
    batch = []
    n_batch_samples = 0
    for db_sentences_path in tqdm(db_sentences_paths, desc='Batching db files'):
        n_samples = cached_count_lines(db_sentences_path)
        if n_batch_samples + n_samples > n_samples_per_gpu:
            db_sentences_paths_batches.append(batch)
            batch = []
            n_batch_samples = 0
        batch.append(db_sentences_path)
        n_batch_samples += n_samples
    if batch:  # Make sure to add the last batch if it's not empty
        db_sentences_paths_batches.append(batch)
    
    # Set up data for parallel jobs
    intermediary_results_paths = []
    offsets = []
    offset = 0
    for db_batch in db_sentences_paths_batches:
        # Generate expected result path for each batch
        result_path = get_results_path(
            query_sentences_path, db_batch, topk, nprobe, nn_search_results_dir
        )
        intermediary_results_paths.append(result_path)
        offsets.append(offset)
        offset += sum([cached_count_lines(sentences_path) for sentences_path in db_batch])
    
    # Process batches that don't already exist
    batches_to_process = []
    for i, (batch, result_path) in enumerate(zip(db_sentences_paths_batches, intermediary_results_paths)):
        if not result_path.exists():
            batches_to_process.append((i, batch, result_path))
    
    if batches_to_process:
        total_batches = len(batches_to_process)
        print(f"Processing {total_batches} batches in parallel across {torch.cuda.device_count()} GPUs")
        
        # Define the function to process one batch with improved GPU handling
        def process_batch(batch_idx, db_batch, result_path):
            batch_start_time = time.time()
            
            # Distribute work across all available GPUs
            gpu_id = batch_idx % torch.cuda.device_count()
            
            # Print detailed debug information
            print(f"Processing batch {batch_idx}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Attempting to use GPU {gpu_id}")
            
            # Use more explicit device selection
            torch.cuda.set_device(gpu_id)
            print(f"Set device to GPU {gpu_id}, current device: {torch.cuda.current_device()}")
            
            try:
                # Load and prepare indexes
                query_index = load_index(get_index_path(query_sentences_path, indexes_dir))
                db_paths = [get_index_path(path, indexes_dir) for path in db_batch]
                
                # Load indexes with explicit GPU configuration
                try:
                    # Try using optimized loading if available
                    res = faiss.StandardGpuResources()
                    config = faiss.GpuIndexFlatConfig()
                    config.useFloat16 = True
                    config.device = gpu_id
                    db_index = load_indexes(db_paths, gpu_id, res, config)
                except TypeError:
                    # Fall back to standard loading if the optimized version isn't implemented
                    db_index = load_indexes(db_paths)
                
                # Get nearest neighbors
                distances, sentence_ids = get_nearest_sentence_ids(
                    query_index, db_index, topk, nprobe
                )
                
                # Save results
                dump_results(distances, sentence_ids, result_path)
                
                batch_time = time.time() - batch_start_time
                print(f"Successfully completed batch {batch_idx} on GPU {gpu_id} in {batch_time:.2f}s")
                return batch_idx, batch_time
                
            except Exception as e:
                print(f"Error processing batch {batch_idx} on GPU {gpu_id}: {e}")
                # Re-raise to ensure job failure is tracked
                raise
        
        # Use ThreadPoolExecutor for more dynamic work distribution
        completed_batches = 0
        batch_times = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_jobs, len(batches_to_process))) as executor:
            # Submit initial batch of jobs
            futures = {}
            for batch_idx, db_batch, result_path in batches_to_process[:n_jobs]:
                future = executor.submit(process_batch, batch_idx, db_batch, result_path)
                futures[future] = (batch_idx, db_batch, result_path)
            
            # Monitor progress and estimate completion time
            progress_bar = tqdm(total=total_batches, desc="Processing batches")
            
            # Process as completed
            while futures:
                # Wait for the next future to complete
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    batch_idx, batch_time = future.result()
                    completed_batches += 1
                    batch_times.append(batch_time)
                    
                    # Update progress bar
                    progress_bar.update(1)
                    
                    # Calculate and display ETA
                    if len(batch_times) >= 3:  # After a few batches, we can estimate time
                        avg_time_per_batch = sum(batch_times) / len(batch_times)
                        remaining_batches = total_batches - completed_batches
                        eta_seconds = avg_time_per_batch * remaining_batches / min(torch.cuda.device_count(), n_jobs)
                        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                        elapsed = str(datetime.timedelta(seconds=int(time.time() - overall_start_time)))
                        
                        progress_bar.set_description(
                            f"Processing batches - {completed_batches}/{total_batches} - "
                            f"Elapsed: {elapsed} - ETA: {eta}"
                        )
                    
                    # Remove the completed future
                    del futures[future]
                    
                    # Submit a new job if there are more to process
                    next_idx = n_jobs + completed_batches - 1
                    if next_idx < total_batches:
                        batch_info = batches_to_process[next_idx]
                        future = executor.submit(process_batch, *batch_info)
                        futures[future] = batch_info
            
            progress_bar.close()
    
    # Check if all results were generated
    missing_results = [path for path in intermediary_results_paths if not path.exists()]
    if missing_results:
        raise RuntimeError(f"Missing {len(missing_results)} expected result files")
    
    # Combine results
    combine_start_time = time.time()
    if len(intermediary_results_paths) == 1:
        # If only one batch, no need to combine
        if combined_results_path != intermediary_results_paths[0]:
            import shutil
            shutil.copy(intermediary_results_paths[0], combined_results_path)
    else:
        # Combine and save final result
        print(f"Combining {len(intermediary_results_paths)} intermediary results...")
        distances, sentence_ids = print_running_time(combine_results_over_db_indexes)(
            intermediary_results_paths, offsets
        )
        dump_results(distances, sentence_ids, combined_results_path)
        
        # Delete intermediary results
        if delete_intermediary:
            deleted_count = 0
            for intermediary_results_path in intermediary_results_paths:
                try:
                    intermediary_results_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Warning: Could not delete {intermediary_results_path}: {e}")
            print(f"Cleaned up {deleted_count}/{len(intermediary_results_paths)} intermediary files")
                
    # Print total time
    total_time = time.time() - overall_start_time
    print(f"Total processing time: {str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"Combined results saved to: {combined_results_path}")
    
    return combined_results_path