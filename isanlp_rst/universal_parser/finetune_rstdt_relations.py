"""Fine-tune the released UniRST relation head on merged native RST-DT labels."""

import json
import pickle
import random
from pathlib import Path

import fire
import numpy as np
import torch

from isanlp_rst.universal_parser.data_manager import DataManager
from isanlp_rst.universal_parser.finetune_gum_relations import _class_weights
from isanlp_rst.universal_parser.predictor import PredictorUniRST
from isanlp_rst.universal_parser.relation_finetuning import (
    configure_dropout,
    freeze_except_relation_head,
    replace_with_rstdt_fine_head,
)
from isanlp_rst.universal_parser.src.parser.data import RelationTableRSTDTFine
from isanlp_rst.universal_parser.src.parser.training_manager import TrainingManager


RELEASED_MODEL = 'tchewik/isanlp_rst_v3'
RELEASED_REVISION = '9407970f1d9d2435b5f875a0cd14293a16646304'
RSTDT_INVENTORY = 'eng.rst.rstdt'


def _load_data_manager(path, data_root):
    path = Path(path)
    requested_root = Path(data_root).expanduser().resolve()
    if path.exists():
        with path.open('rb') as stream:
            manager = pickle.load(stream)
        is_current = getattr(manager, 'split_source', '').startswith(
            'https://github.com/disrpt/sharedtask2025')
        if not is_current:
            manager = DataManager(
                'RST-DT', relation_granularity='fine', data_root=requested_root)
            manager.from_rs3()
            manager.save(path)
        else:
            manager.data_root = requested_root
            manager.input_path = requested_root / 'rstdt_rs3_merged'
            manager.output_path = requested_root / 'rstdt_fine_prepared'
            list_root = requested_root / 'rstdt_file_lists'
            manager.corpus = {
                part: (list_root / f'files.{part}').read_text(
                    encoding='utf8').splitlines()
                for part in ('train', 'dev', 'test')
            }
            manager._validate_rstdt_partitions()
        if not manager.output_path.is_dir() or not all(
            manager.output_path.joinpath(name + '.pkl').is_file()
            for names in manager.corpus.values() for name in names
        ):
            raise FileNotFoundError(
                f'Cached manager {path} was rebased to {manager.output_path}, but no '
                'prepared RST-DT documents were found there. Rebuild the cache for '
                'this --data-root.'
            )
    else:
        manager = DataManager(
            'RST-DT', relation_granularity='fine',
            data_root=requested_root)
        manager.from_rs3()
        path.parent.mkdir(parents=True, exist_ok=True)
        manager.save(path)
    if (
        getattr(manager, 'corpus_name', None) != 'RST-DT'
        or getattr(manager, 'relation_granularity', None) != 'fine'
        or list(getattr(manager, 'relation_table', ())) != list(RelationTableRSTDTFine)
    ):
        raise ValueError(f'{path} is not a fine-grained RST-DT data manager')
    return manager


def train(
    data_manager_path='data/data_manager_rstdt_fine.pickle',
    data_root='data',
    save_dir='saves',
    run_name='rstdt_merged_fine_relations',
    model_name=RELEASED_MODEL,
    revision=RELEASED_REVISION,
    cuda_device=-1,
    batch_size=1,
    eval_size=1,
    epochs=20,
    lr=1e-4,
    weight_decay=1e-2,
    patience=4,
    grad_norm=1.0,
    grad_clipping_value=10.0,
    seed=42,
    use_amp=False,
    class_weight_power=0.5,
    class_weight_smoothing=1.0,
    fine_tune_scope='relation_head',
    transformer_lr_multiplier=0.1,
    classifier_dropout=None,
    transformer_dropout=None,
    segmenter_dropout=None,
    evaluate_test_after_training=True,
):
    """Fine-tune UniRST with the 110-way suffix-merged RST-DT inventory."""

    if fine_tune_scope not in {'relation_head', 'all'}:
        raise ValueError("fine_tune_scope must be either 'relation_head' or 'all'")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_device >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError('cuda_device was requested, but CUDA is unavailable')
        torch.cuda.manual_seed_all(seed)

    manager = _load_data_manager(data_manager_path, data_root)
    train_data, dev_data, test_data = manager.get_data()
    predictor = PredictorUniRST(
        hf_model_name=model_name,
        hf_model_version=revision,
        relinventory=RSTDT_INVENTORY,
        cuda_device=cuda_device,
    )
    replace_with_rstdt_fine_head(predictor.model)
    effective_dropout = configure_dropout(
        predictor.model,
        classifier=classifier_dropout,
        transformer=transformer_dropout,
        segmenter=segmenter_dropout,
    )
    if fine_tune_scope == 'relation_head':
        freeze_except_relation_head(predictor.model)
    else:
        predictor.model.requires_grad_(True)
    predictor.label_maps = None

    train_data = predictor.tokenize(train_data)
    dev_data = predictor.tokenize(dev_data)
    test_data = predictor.tokenize(test_data)
    relation_weights, relation_counts = _class_weights(
        train_data, len(RelationTableRSTDTFine), predictor._cuda_device,
        power=class_weight_power, smoothing=class_weight_smoothing)
    predictor.model.label_weights = [relation_weights]

    provenance = {
        'task': 'rstdt_fine_relation_head',
        'base_model': model_name,
        'base_revision': revision,
        'source_data': 'data/rstdt_rs3_merged',
        'relation_inventory': list(RelationTableRSTDTFine),
        'selection_metric': 'gs_val_f1_rel',
        'test_policy': (
            'evaluated once after validation-selected training'
            if evaluate_test_after_training else 'not evaluated'),
        'seed': seed,
        'artifact_format': 1,
        'fine_tune_scope': fine_tune_scope,
        'dropout': effective_dropout,
        'training_objective': (
            'relation' if fine_tune_scope == 'relation_head'
            else 'tree+relation+segmentation'),
        'class_weighting': {
            'formula': '(count + smoothing) ** (-power), normalized to mean 1',
            'power': class_weight_power,
            'smoothing': class_weight_smoothing,
            'counts': relation_counts.tolist(),
            'weights': relation_weights.detach().cpu().tolist(),
        },
    }
    trainer = TrainingManager(
        predictor.model, [train_data], [dev_data], [test_data],
        batch_size=batch_size, eval_size=eval_size, epochs=epochs, lr=lr,
        transformer_lr_multiplier=(
            transformer_lr_multiplier if fine_tune_scope == 'all' else 0),
        lr_decay_epoch=1000, lr_decay=1.0, weight_decay=weight_decay,
        grad_norm=grad_norm, grad_clipping_value=grad_clipping_value,
        patience=patience, use_micro_f1=True, use_dwa_loss=False,
        dwa_bs=max(batch_size, 1), save_dir=save_dir, use_amp=use_amp,
        run_name=run_name, config=provenance,
        loss_mode='relation' if fine_tune_scope == 'relation_head' else 'all',
        selection_metric='gs_val_f1_rel', evaluate_test_each_epoch=False,
        checkpoint_scope=(
            'relation_head' if fine_tune_scope == 'relation_head' else 'full'),
    )
    run_dir = trainer.save_dir
    (run_dir / 'relation_table_eng.rst.rstdt.txt').write_text(
        '\n'.join(RelationTableRSTDTFine) + '\n', encoding='utf8')
    best_metrics = trainer.train()
    test_metrics = None
    if evaluate_test_after_training:
        if fine_tune_scope == 'relation_head':
            predictor.model.label_classifier.load_state_dict(torch.load(
                run_dir / 'best_relation_head.pt', map_location=predictor._cuda_device,
                weights_only=True))
        else:
            predictor.model.load_state_dict(torch.load(
                run_dir / 'best_weights.pt', map_location=predictor._cuda_device,
                weights_only=True))
        test_metrics = trainer.evaluate_test()
        with (run_dir / 'test_metrics.json').open('w', encoding='utf8') as stream:
            json.dump(test_metrics, stream, indent=2, sort_keys=True, default=float)
    return {'run_dir': str(run_dir), 'best_validation': best_metrics, 'test': test_metrics}


if __name__ == '__main__':
    fire.Fire(train)
