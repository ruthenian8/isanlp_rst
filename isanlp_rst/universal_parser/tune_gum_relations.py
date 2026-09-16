"""Small validation-only hyperparameter sweep for GUM fine relations."""

import gc
import json
from pathlib import Path

import fire
import torch

from isanlp_rst.universal_parser.finetune_gum_relations import train


CANDIDATES = (
    {
        'name': 'unweighted_lr3e5',
        'lr': 3e-5,
        'weight_decay': 1e-2,
        'class_weight_power': 0.0,
        'class_weight_smoothing': 10.0,
    },
    {
        'name': 'moderate_lr3e5',
        'lr': 3e-5,
        'weight_decay': 1e-2,
        'class_weight_power': 0.25,
        'class_weight_smoothing': 10.0,
    },
    {
        'name': 'moderate_lr1e4',
        'lr': 1e-4,
        'weight_decay': 1e-2,
        'class_weight_power': 0.25,
        'class_weight_smoothing': 10.0,
    },
    {
        'name': 'moderate_wd5e2',
        'lr': 3e-5,
        'weight_decay': 5e-2,
        'class_weight_power': 0.25,
        'class_weight_smoothing': 10.0,
    },
)

FULL_MODEL_DROPOUT_CANDIDATES = (
    {
        'name': 'moderate_transformer_dropout',
        'lr': 3e-5,
        'weight_decay': 1e-2,
        'class_weight_power': 0.25,
        'class_weight_smoothing': 10.0,
        'transformer_dropout': 0.2,
    },
    {
        'name': 'moderate_classifier_dropout',
        'lr': 3e-5,
        'weight_decay': 1e-2,
        'class_weight_power': 0.25,
        'class_weight_smoothing': 10.0,
        'classifier_dropout': 0.6,
    },
)


def tune(
    data_manager_path='data/data_manager_gum_fine.pickle',
    data_root='data',
    save_dir='saves/gum_tuning',
    cuda_device=-1,
    fine_tune_scope='relation_head',
    batch_size=1,
    eval_size=1,
    epochs=10,
    patience=3,
    seed=42,
    use_amp=False,
):
    """Run conservative imbalance/LR candidates, selecting on dev only."""

    results = []
    candidates = CANDIDATES
    if fine_tune_scope == 'all':
        candidates += FULL_MODEL_DROPOUT_CANDIDATES
    for candidate in candidates:
        run_name = f"{fine_tune_scope}_{candidate['name']}_seed{seed}"
        result = train(
            data_manager_path=data_manager_path,
            data_root=data_root,
            save_dir=save_dir,
            run_name=run_name,
            cuda_device=cuda_device,
            batch_size=batch_size,
            eval_size=eval_size,
            epochs=epochs,
            patience=patience,
            seed=seed,
            use_amp=use_amp,
            fine_tune_scope=fine_tune_scope,
            evaluate_test_after_training=False,
            **{key: value for key, value in candidate.items() if key != 'name'},
        )
        score = float(result['best_validation']['gs_val_f1_rel'])
        results.append({
            'name': candidate['name'],
            'run_dir': result['run_dir'],
            'gs_val_f1_rel': score,
            'parameters': {key: value for key, value in candidate.items() if key != 'name'},
        })
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.sort(key=lambda item: item['gs_val_f1_rel'], reverse=True)
    summary = {
        'selection_metric': 'gs_val_f1_rel',
        'test_evaluated': False,
        'best': results[0],
        'results': results,
    }
    summary_path = Path(save_dir) / f'tuning_summary_{fine_tune_scope}_seed{seed}.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf8') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    return summary


if __name__ == '__main__':
    fire.Fire(tune)
