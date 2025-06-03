from .base_eval_set import BaseEvalSet
from pathlib import Path


class GeneralEvalSet(BaseEvalSet):
    def __init__(
        self, general_benchmark_path: Path, chroma_db_path: Path | None = None
    ):
        self._general_benchmark_path = general_benchmark_path

        _questions_df_path = self._general_benchmark_path / "questions_df.csv"
        _corpora_dir_path = self._general_benchmark_path / "corpora"
        _corpora_filenames = [f for f in _corpora_dir_path.iterdir() if f.is_file()]
        corpora_id_paths = {f.stem: str(f) for f in _corpora_filenames}

        if chroma_db_path is not None:
            super().__init__(
                str(_questions_df_path),
                str(chroma_db_path),
                corpora_id_paths=corpora_id_paths,
            )
        else:
            super().__init__(
                str(_questions_df_path),
                corpora_id_paths=corpora_id_paths,
            )
        self.is_general = True
