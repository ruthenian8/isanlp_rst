"""Compare single-document and batched UniRST inference on a CUDA device."""

import argparse
import json
import time
from pathlib import Path

import torch

from .predictor import PredictorUniRST
from isanlp_rst.utils import tree_to_dict


def compare_encodings(reference, actual, atol, rtol):
    maximum = 0.0
    close = len(reference) == len(actual)
    for (expected_breaks, expected_tensors), (breaks, tensors) in zip(reference, actual):
        close &= expected_breaks == breaks
        for expected, tensor in zip(expected_tensors, tensors):
            if expected.shape != tensor.shape:
                close = False
                continue
            error = (expected - tensor).abs().max().item()
            maximum = max(maximum, error)
            close &= bool(torch.isfinite(tensor).all()) and torch.allclose(
                expected, tensor, atol=atol, rtol=rtol)
    return {'allclose': bool(close), 'max_absolute_error': maximum}


def capture_encoder(predictor, run):
    records = []

    def capture(module, inputs, output):
        encodings, hidden, _, breaks = output[:4]
        for i, boundaries in enumerate(breaks):
            records.append((list(boundaries), (
                encodings[i, :len(boundaries)].detach().float().cpu().clone(),
                hidden[:, i].detach().float().cpu().clone(),
            )))

    handle = predictor.model.encoder.register_forward_hook(capture)
    try:
        results = run()
    finally:
        handle.remove()
    return results, records


def tree_snapshot(node):
    result = tree_to_dict(node)
    pending = [(node, result)] if node is not None else []
    while pending:
        node, record = pending.pop()
        record['entropy'] = getattr(node, 'entropy', None)
        for side in ('left', 'right'):
            child = getattr(node, side, None)
            if child is not None:
                pending.append((child, record[side]))
    return result


def structure(tree):
    return {key: structure(value) if isinstance(value, dict) else value
            for key, value in tree.items() if key not in {'entropy', 'proba'}}


def measure(run, device, repeats, count):
    run()  # Warm up this batch shape before timing.
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(repeats):
        run()
    torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - started) / repeats
    return {'seconds_per_pass': seconds, 'documents_per_second': count / seconds,
            'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path, help='JSON array of document strings; otherwise use built-in examples')
    cli.add_argument('--from-edus', action='store_true',
                     help='Read a JSON array of EDU arrays and benchmark supplied boundaries')
    source = cli.add_mutually_exclusive_group()
    source.add_argument('--model-dir', help='Local released UniRST model directory')
    source.add_argument('--relation-head-dir', help='Fine-tuned artifact directory (head or full model)')
    cli.add_argument('--model-name', default='tchewik/isanlp_rst_v3')
    cli.add_argument('--revision', default='9407970f1d9d2435b5f875a0cd14293a16646304')
    cli.add_argument('--relinventory', default='eng.rst.rstdt')
    cli.add_argument('--device', type=int, default=0)
    cli.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8])
    cli.add_argument('--repeats', type=int, default=2)
    cli.add_argument('--atol', type=float, default=1e-5)
    cli.add_argument('--rtol', type=float, default=1e-4)
    args = cli.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.batch_sizes):
        cli.error('repeats and batch sizes must be positive')
    if args.atol < 0 or args.rtol < 0:
        cli.error('tolerances must be nonnegative')
    if not torch.cuda.is_available() or not 0 <= args.device < torch.cuda.device_count():
        cli.error('a valid CUDA device is required')

    sentence = 'The experiment finished successfully. However, further evidence is needed before drawing conclusions.'
    examples = [
        '', 'Hello.', 'Two words', sentence, sentence * 2, sentence * 4,
        (sentence + ' ') * 45, (sentence + ' ') * 60,
    ]
    texts = json.loads(args.input.read_text(encoding='utf8')) if args.input else (
        [['Hello.'], ['Two words']] + [[sentence] * count for count in (2, 3, 4, 45, 60)]
        if args.from_edus else examples)
    if args.from_edus:
        if not isinstance(texts, list) or not texts:
            cli.error('input must be a nonempty JSON array of EDU arrays')
        if any(not isinstance(document, list) or not document or
               any(not isinstance(edu, str) or not edu for edu in document)
               for document in texts):
            cli.error('each document must be a nonempty array of nonempty EDU strings')
        if not any(len(document) > 1 for document in texts):
            cli.error('include at least one document with multiple EDUs to exercise the model')
    else:
        if not isinstance(texts, list) or not texts or any(not isinstance(text, str) for text in texts):
            cli.error('input must be a nonempty JSON array of strings')
        if not any(len(text.split()) >= 3 for text in texts):
            cli.error('include at least one document with three or more words to exercise the model')

    torch.manual_seed(42)
    torch.cuda.set_device(args.device)
    if args.relation_head_dir:
        predictor = PredictorUniRST.from_fine_relation_head(args.relation_head_dir, cuda_device=args.device)
    else:
        predictor = PredictorUniRST(
            model_dir=args.model_dir,
            hf_model_name=None if args.model_dir else args.model_name,
            hf_model_version=args.revision, relinventory=args.relinventory,
            cuda_device=args.device)
    predictor.model.eval()
    parse_one = predictor.parse_from_edus if args.from_edus else predictor.parse_rst
    parse_batch = predictor.parse_from_edus_batch if args.from_edus else predictor.parse_batch
    single = lambda: [parse_one(text) for text in texts]
    baseline, encodings = capture_encoder(predictor, single)
    snapshots = [tree_snapshot(result['rst'][0]) for result in baseline]
    timing = measure(single, args.device, args.repeats, len(texts))
    report = {
        'gpu': torch.cuda.get_device_name(args.device), 'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'model': args.model_dir or args.relation_head_dir or args.model_name,
        'revision': args.revision if not (args.model_dir or args.relation_head_dir) else None,
        'inventory': predictor.relinventory, 'documents': len(texts),
        'repeats': args.repeats, 'atol': args.atol, 'rtol': args.rtol,
        'from_edus': args.from_edus,
        'baseline': timing, 'batches': [],
    }
    passed = True
    for size in args.batch_sizes:
        run = lambda: parse_batch(texts, batch_size=size)
        results, actual = capture_encoder(predictor, run)
        trees = [tree_snapshot(result['rst'][0]) for result in results]
        mismatches = [i for i, (expected, tree) in enumerate(zip(snapshots, trees))
                      if structure(expected) != structure(tree)]
        comparison = compare_encodings(encodings, actual, args.atol, args.rtol)
        metrics = measure(run, args.device, args.repeats, len(texts))
        metrics.update(batch_size=size, structure_mismatch_document_indices=mismatches,
                       exact_tree_matches=sum(a == b for a, b in zip(snapshots, trees)),
                       encoder=comparison,
                       speedup=timing['seconds_per_pass'] / metrics['seconds_per_pass'])
        report['batches'].append(metrics)
        passed &= not mismatches and len(trees) == len(snapshots) and comparison['allclose']
    report['passed'] = bool(passed)
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
