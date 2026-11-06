from typing import Sequence

from .dmrst_parser.predictor import PredictorDMRST
from .universal_parser.predictor import PredictorUniRST


class Parser:
    DMRST_PARSERS = ('gumrrg', 'rstdt', 'rstreebank')
    UNIVERSAL_PARSERS = ('rrtrrg', 'unirst')
    AVAILABLE_VERSIONS = DMRST_PARSERS + UNIVERSAL_PARSERS

    def __init__(self,
                 model_dir: str = None,
                 hf_model_name: str = 'tchewik/isanlp_rst_v3',
                 hf_model_version: str = None,
                 relinventory: str = None,   # for universal parsers
                 relinventory_idx: int = 0,  # for universal parsers
                 relation_head_dir: str = None,
                 cuda_device: int = -1):

        if relation_head_dir is not None:
            if model_dir is not None:
                raise ValueError('model_dir and relation_head_dir cannot be used together')
            self.predictor = PredictorUniRST.from_fine_relation_head(
                relation_head_dir, cuda_device=cuda_device)
        elif hf_model_version in self.DMRST_PARSERS:
            self.predictor = PredictorDMRST(
                model_dir=model_dir,
                hf_model_name=hf_model_name,
                hf_model_version=hf_model_version,
                cuda_device=cuda_device
            )
        elif hf_model_version in self.UNIVERSAL_PARSERS:
            self.predictor = PredictorUniRST(
                model_dir=model_dir,
                hf_model_name=hf_model_name,
                hf_model_version=hf_model_version,
                relinventory=relinventory,
                relinventory_idx=relinventory_idx,
                cuda_device=cuda_device
            )
        else:
            raise NotImplementedError(
                f"Available options for hf_model_version are: {', '.join(self.AVAILABLE_VERSIONS)}"
            )

    def __call__(self, text: str):
        return self.predictor.parse_rst(text)

    def parse_batch(self, texts: Sequence[str], batch_size: int = 8):
        """Parse multiple documents using the UniRST batch path."""
        if not hasattr(self.predictor, 'parse_batch'):
            raise NotImplementedError('Batch parsing is available for UniRST only')
        return self.predictor.parse_batch(texts, batch_size=batch_size)

    def from_edus(self, edus: Sequence[str]):
        """Parse a document using predefined EDUs."""

        return self.predictor.parse_from_edus(edus)

    def from_edus_batch(self, documents: Sequence[Sequence[str]], batch_size: int = 8):
        """Parse multiple documents with supplied EDU boundaries using UniRST."""
        if not hasattr(self.predictor, 'parse_from_edus_batch'):
            raise NotImplementedError('Batch parsing is available for UniRST only')
        return self.predictor.parse_from_edus_batch(documents, batch_size=batch_size)
