"""Output-preservation checks for the inference execution optimizations."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from isanlp_rst.universal_parser.src.parser.modules import DecoderRNN, DefaultLabelClassifier, EncoderRNN
from isanlp_rst.universal_parser.src.parser.parsing_net import ParsingNet
from isanlp_rst.universal_parser.src.parser.segmenters import CRF, ToNySegmenter
from isanlp_rst.utils.du_converter import DUConverter
from isanlp_rst.universal_parser.predictor import PredictorUniRST
from isanlp_rst.universal_parser.benchmark_batch import compare_encodings, structure, tree_snapshot


DEVICES = [
    'cpu',
    pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA unavailable')),
]


def test_benchmark_detects_numeric_and_boundary_disagreements():
    reference = [([1, 3], (torch.ones(2, 4), torch.ones(1, 4)))]
    assert compare_encodings(reference, reference, 1e-5, 1e-4)['allclose']
    perturbed = [([1, 3], (torch.ones(2, 4) + 0.01, torch.ones(1, 4)))]
    result = compare_encodings(reference, perturbed, 1e-5, 1e-4)
    assert not result['allclose']
    assert result['max_absolute_error'] == pytest.approx(0.01)
    assert not compare_encodings(reference, [], 1e-5, 1e-4)['allclose']
    wrong_breaks = [([0, 3], reference[0][1])]
    assert not compare_encodings(reference, wrong_breaks, 1e-5, 1e-4)['allclose']
    wrong_shape = [([1, 3], (torch.ones(3, 4), torch.ones(1, 4)))]
    assert not compare_encodings(reference, wrong_shape, 1e-5, 1e-4)['allclose']


def test_benchmark_distinguishes_structure_from_entropy():
    first = tree_snapshot(SimpleNamespace(text='Example', entropy=0.1))
    second = tree_snapshot(SimpleNamespace(text='Example', entropy=0.2))
    assert first != second
    assert structure(first) == structure(second)
    second['text'] = 'Different'
    assert structure(first) != structure(second)


@pytest.mark.parametrize('device', DEVICES)
def test_short_document_transformer_batch_matches_individual_inference(device):
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(pad_token_id=0)
            self.embedding = nn.Embedding(20, 4)
            self.calls = 0

        def forward(self, ids, attention_mask=None):
            self.calls += 1
            return (self.embedding(ids),)

    class Segmenter(nn.Module):
        def test_segment_loss(self, embeddings, sent_breaks):
            return [len(embeddings) - 1]

    torch.manual_seed(2)
    transformer = Transformer()
    encoder = EncoderRNN(transformer, 4, 4, 1, 0, False,
                         nn.ModuleList([Segmenter()]), 'avg', True, False, 1,
                         256, 64, token_bilstm_hidden=0,
                         cuda_device=torch.device(device)).to(device).eval()
    documents = [[1, 2, 3], [4, 5, 6, 7, 8]]
    with torch.inference_mode():
        singles = [encoder([doc], None, None, [[]], is_test=True, dataset_index=[0])
                   for doc in documents]
        transformer.calls = 0
        batched = encoder(documents, None, None, [[], []], is_test=True,
                          dataset_index=[0, 0])
    assert transformer.calls == 1
    assert batched[3] == [result[3][0] for result in singles]
    for i, result in enumerate(singles):
        torch.testing.assert_close(batched[0][i], result[0][0])
        torch.testing.assert_close(batched[1][:, i], result[1][:, 0])


def test_batch_parser_preserves_order_and_single_edu_results():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {'input_ids': [[1] * len(text.split()) for text in texts],
                    'offset_mapping': [[(i * 2, i * 2 + 1) for i in range(len(text.split()))]
                                       for text in texts]}

        def convert_ids_to_tokens(self, ids):
            return ['▁a'] * len(ids)

    class Model:
        def testing_loss(self, sentences, *args, **kwargs):
            return 0, 0, [['NONE'] for _ in sentences], None, [
                [len(sentence) - 1] for sentence in sentences]

    predictor = PredictorUniRST.__new__(PredictorUniRST)
    predictor.tokenizer = Tokenizer()
    predictor.model = Model()
    predictor.label_maps = None
    predictor.relinventory_idx = 0
    texts = ['a b c', 'a', 'a b c d', 'a b c']
    expected = [predictor.parse_rst(text) for text in texts]
    actual = predictor.parse_batch(texts, batch_size=2)
    assert [result['rst'][0].text for result in actual] == [
        result['rst'][0].text for result in expected]
    with pytest.raises(ValueError, match='positive'):
        predictor.parse_batch(texts, batch_size=0)
    with pytest.raises(TypeError, match='sequence'):
        predictor.parse_batch('a b c')
    with pytest.raises(TypeError, match='integer'):
        predictor.parse_batch(texts, batch_size=1.5)
    assert predictor.parse_batch([]) == []


@pytest.mark.parametrize('device', DEVICES)
def test_batch_attention_and_trees_match_single_documents(device):
    from transformers import XLMRobertaConfig, XLMRobertaModel
    from isanlp_rst.utils import tree_to_dict

    class Segmenter(nn.Module):
        def test_segment_loss(self, embeddings, sent_breaks):
            return list(range(len(embeddings)))

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {'input_ids': [[3] * len(text.split()) for text in texts],
                    'offset_mapping': [[(i * 2, i * 2 + 1) for i in range(len(text.split()))]
                                       for text in texts]}

        def convert_ids_to_tokens(self, ids):
            return ['▁a'] * len(ids)

    torch.manual_seed(8)
    transformer = XLMRobertaModel(XLMRobertaConfig(
        vocab_size=20, max_position_embeddings=514, hidden_size=4,
        num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=8, hidden_dropout_prob=0, attention_probs_dropout_prob=0))
    model = _parser(device, True)
    model.encoder = EncoderRNN(
        transformer, 4, 4, 1, 0, True, nn.ModuleList([Segmenter()]), 'avg',
        True, False, 1, 256, 64, token_bilstm_hidden=0,
        cuda_device=torch.device(device)).to(device).eval()
    predictor = PredictorUniRST.__new__(PredictorUniRST)
    predictor.tokenizer = Tokenizer()
    predictor.model = model
    predictor.label_maps = None
    predictor.relinventory_idx = 0
    texts = ['a b c', 'a', 'a b c d e', '', 'a b c d']
    keys = tuple(model.state_dict())
    expected = [tree_to_dict(predictor.parse_rst(text)['rst'][0]) for text in texts]
    for size in (1, 2, 8):
        actual = [tree_to_dict(result['rst'][0])
                  for result in predictor.parse_batch(texts, batch_size=size)]
        assert actual == expected
    assert tuple(model.state_dict()) == keys

    docs = [[3, 4, 5], [6, 7, 8, 9, 10]] + [[3] * size for size in (511, 512, 513)]
    with torch.inference_mode():
        singles = [model.encoder([doc], None, None, [[]], is_test=True, dataset_index=[0])
                   for doc in docs]
        batched = model.encoder(docs, None, None, [[] for _ in docs], is_test=True,
                                dataset_index=[0] * len(docs))
    for i, (outputs, hidden, _, _) in enumerate(singles):
        torch.testing.assert_close(batched[0][i, :len(docs[i])], outputs[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(batched[1][:, i], hidden[:, 0], atol=1e-6, rtol=1e-5)


def _legacy_viterbi(crf, emissions, mask):
    """Original recurrence and device-side traceback, including padding."""
    length, batch, _ = emissions.shape
    score = crf.start_transitions + emissions[0]
    history = []
    for step in range(1, length):
        next_score = score.unsqueeze(2) + crf.transitions + emissions[step].unsqueeze(1)
        next_score, indices = next_score.max(dim=1)
        score = torch.where(mask[step].unsqueeze(1).bool(), next_score, score)
        history.append(indices)
    score += crf.end_transitions
    ends = mask.long().sum(dim=0) - 1
    result = []
    for idx in range(batch):
        tag = score[idx].max(0)[1]
        tags = [torch.zeros_like(tag) for _ in range(length - ends[idx])]
        tags.append(tag)
        for previous in reversed(history[:ends[idx]]):
            tag = previous[idx][tags[-1]]
            tags.append(tag)
        result.append(torch.stack(list(reversed(tags))))
    return torch.stack(result)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('length', [1, 7])
@pytest.mark.parametrize('tied', [False, True])
def test_crf_traceback_exactly_matches_original(device, length, tied):
    torch.manual_seed(41)
    crf = CRF(3).to(device)
    emissions = torch.randn(length, 3, 3, device=device)
    lengths = torch.tensor([length, max(1, length - 2), 1], device=device)
    mask = torch.arange(length, device=device)[:, None] < lengths
    if tied:
        emissions.zero_()
        for parameter in crf.parameters():
            nn.init.zeros_(parameter)
    state_keys = tuple(crf.state_dict())

    with torch.inference_mode():
        expected = _legacy_viterbi(crf, emissions, mask)
        actual = crf._viterbi_decode(emissions, mask)

    assert torch.equal(actual, expected)
    assert actual.device == emissions.device
    assert actual.shape == (3, length + 1)
    assert tuple(crf.state_dict()) == state_keys


@pytest.mark.parametrize('device', DEVICES)
def test_segment_boundaries_match_original_crf_output(device):
    torch.manual_seed(12)
    segmenter = ToNySegmenter(
        4, use_lstm=False, use_crf=True, cuda_device=torch.device(device)).eval()
    embeddings = torch.randn(19, 4, device=device)
    with torch.inference_mode():
        emissions = segmenter(segmenter.encode(embeddings)).unsqueeze(1)
        mask = torch.ones(19, 1, dtype=torch.bool, device=device)
        tags = _legacy_viterbi(segmenter.crf, emissions, mask)[0].cpu().tolist()
        expected = [idx for idx, tag in enumerate(tags) if tag == 1]
        if 18 not in expected:
            expected.append(18)
        actual = segmenter.test_segment_loss(embeddings)
    assert actual == expected


class _Encoder(nn.Module):
    def forward(self, sentences, entities, positions, breaks, **kwargs):
        device = self.encodings.device
        count = max(map(len, breaks))
        outputs = self.encodings[:count].unsqueeze(0).expand(len(breaks), -1, -1)
        hidden = torch.zeros(1, len(breaks), 4, device=device)
        return outputs, hidden, 0, breaks

    def __init__(self, device):
        super().__init__()
        self.register_buffer('encodings', torch.arange(24, device=device).reshape(6, 4).float() / 10)


class _Pointer(nn.Module):
    def forward(self, encodings, state):
        logits = torch.arange(len(encodings), device=encodings.device).float() / 7
        logits[1 if len(encodings) > 2 else 0] += 1
        return logits.softmax(0).unsqueeze(0), logits.log_softmax(0).unsqueeze(0)


def _parser(device, union):
    torch.manual_seed(6)
    model = ParsingNet.__new__(ParsingNet)
    nn.Module.__init__(model)
    model._cuda_device = torch.device(device)
    model.du_encoding_kind = 'avg'
    model.rel_classification_kind = 'default'
    model.dataset2classifier = [0]
    model.relation_vocab = ['background_NS']
    model.relation_tables = [model.relation_vocab]
    model.dataset_masks = [torch.ones(1, dtype=torch.bool, device=device)] if union else None
    classifier = DefaultLabelClassifier(4, 3, 1, cuda_device=model._cuda_device)
    if union:
        model.label_classifier = classifier
    else:
        model.label_classifiers = nn.ModuleList([classifier])
    model.encoder = _Encoder(device)
    model.decoder = DecoderRNN(4, 4, 1, 0, model._cuda_device)
    model.pointer = _Pointer()
    return model.eval()


def _legacy_entropy(log_weights):
    count = log_weights.shape[-1]
    if count <= 1:
        return 0.0
    probabilities = torch.exp(log_weights.detach())
    entropy = -torch.sum(probabilities * log_weights.detach(), dim=-1)
    maximum = torch.log(torch.tensor(count, device=log_weights.device, dtype=log_weights.dtype))
    return (entropy / maximum).cpu().item()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('union', [False, True])
def test_decoding_preserves_span_order_entropy_and_return_contract(device, union):
    model = _parser(device, union)
    breaks = [list(range(count)) for count in (6, 3, 2, 1)]
    with torch.inference_mode():
        result = model.testing_loss(
            breaks, None, None, None, breaks, [[], [], [], []], [[], [], [], []],
            generate_tree=True, use_pred_segmentation=False, dataset_index=[0, 0, 0, 0])
        large_entropy = _legacy_entropy(model.pointer(model.encoder.encodings[:5], None)[1])
        small_entropy = _legacy_entropy(model.pointer(model.encoder.encodings[:3], None)[1])
        pair_entropy = _legacy_entropy(model.pointer(model.encoder.encodings[:2], None)[1])
    assert result[2] == [
        [' '.join([
            f'(1:Nucleus=span;entropy={large_entropy:.5f}:2,3:Satellite=background:6)',
            '(1:Nucleus=span;entropy=0.00000:1,2:Satellite=background:2)',
            f'(3:Nucleus=span;entropy={small_entropy:.5f}:4,5:Satellite=background:6)',
            '(3:Nucleus=span;entropy=0.00000:3,4:Satellite=background:4)',
            '(5:Nucleus=span;entropy=0.00000:5,6:Satellite=background:6)',
        ])],
        [f'(1:Nucleus=span;entropy={pair_entropy:.5f}:1,2:Satellite=background:3) '
         '(2:Nucleus=span;entropy=0.00000:2,3:Satellite=background:3)'],
        ['(1:Nucleus=span;entropy=0.00000:1,2:Satellite=background:2)'],
        ['NONE'],
    ]
    assert result[3] == ([0] * 8, [0] * 8)
    assert result[4] == breaks
    assert result[0].item() == result[1].item() == 0.0
    with torch.inference_mode():
        without_tree = model.testing_loss(
            breaks, None, None, None, breaks, [[], [], [], []], [[], [], [], []],
            generate_tree=False, use_pred_segmentation=False, dataset_index=[0, 0, 0, 0])
    assert without_tree[2] is None
    assert without_tree[3:] == result[3:]


@pytest.mark.parametrize('device', DEVICES)
def test_deferred_entropy_preserves_original_scalar_and_formatting(device):
    torch.manual_seed(7)
    model = _parser(device, True)
    distributions = [torch.randn(1, length, device=device).log_softmax(-1)
                     for length in (1, 2, 3, 17, 300)]
    expected = [_legacy_entropy(weights) for weights in distributions]
    tensors = [model._calculate_normalized_entropy(weights, return_tensor=True)
               for weights in distributions]
    actual = torch.stack(tensors).cpu().tolist()
    assert actual == expected
    assert [f'{value:.5f}' for value in actual] == [f'{value:.5f}' for value in expected]
    assert [model._calculate_normalized_entropy(weights) for weights in distributions] == expected


class _LegacyConverter(DUConverter):
    def construct_tree(self, root, edus, rels):
        left_start, left_end, relation, nuclearity, right_start, right_end, entropy = rels[root]
        left = edus[left_start] if left_start == left_end else self.construct_tree(
            self._get_child(left_start, left_end, rels), edus, rels)
        right = edus[right_start] if right_start == right_end else self.construct_tree(
            self._get_child(right_start, right_end, rels), edus, rels)
        self.du_id += 1
        from isanlp.annotation_rst import DiscourseUnit
        return DiscourseUnit(
            id=self.du_id, left=left, right=right, entropy=entropy,
            relation=relation, nuclearity=nuclearity, start=left.start,
            end=right.end, text=left.text + ' ' + right.text)


def _tree_fields(node):
    result = dict(vars(node))
    for child in ('left', 'right'):
        if result.get(child) is not None:
            result[child] = _tree_fields(result[child])
    return result


def test_indexed_tree_construction_preserves_all_fields_and_first_match():
    edus = [SimpleNamespace(start=idx * 2, end=idx * 2 + 1, text=str(idx), id=idx)
            for idx in range(4)]
    rels = [
        (0, 1, 'background', 'NS', 2, 3, 0.12345),
        (2, 2, 'joint-mn', 'NN', 3, 3, 0.0),
        (0, 0, 'cause', 'SN', 1, 1, 0.99999),
        (0, 0, 'unused-duplicate', 'NN', 1, 1, 0.0),
    ]
    legacy = _LegacyConverter({})
    optimized = DUConverter({})
    legacy.du_id = optimized.du_id = len(edus)
    expected = legacy.construct_tree(0, edus, rels)
    actual = optimized.construct_tree(0, edus, rels)
    assert _tree_fields(actual) == _tree_fields(expected)
    assert optimized.du_id == legacy.du_id
