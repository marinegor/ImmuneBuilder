"""Generate nanobody structures from a parquet dataset.

Steps:
1. Number sequences using immunum (IMGT scheme)
2. Remove duplicates on (fr1, fr2, fr3, fr4)
3. Sample 10,000 random sequences and save to <timestamp>.parquet
4. Generate structures using NanoBodyBuilder2, save as <sequence>.pdb.gz

When CUDA is available, prediction is parallelized across GPU workers using Ray.
"""

from __future__ import annotations

import gzip
import os
from collections import abc
from itertools import cycle
from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer
from loguru import logger
from tqdm import tqdm

from ImmuneBuilder.NanoBodyBuilder2 import Nanobody, embed_dim, model_urls
from ImmuneBuilder.models import StructureModule
from ImmuneBuilder.sequence_checks import number_single_sequence
from ImmuneBuilder.util import are_weights_ready, download_file, get_encoding

app = typer.Typer()

CUDA_AVAILABLE = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------


def segment_and_deduplicate(path: str, sequence_col: str) -> pl.DataFrame:
    """Segment all sequences and deduplicate on frameworks in a single lazy pipeline."""
    import immunum.polars as imp

    return (
        pl.scan_parquet(path)
        .filter(pl.col(sequence_col).str.len_chars() >= 100)
        .with_columns(
            imp.segment(pl.col(sequence_col), chains=["H"], scheme="imgt").alias(
                "segmented"
            )
        )
        .with_columns(
            pl.col("segmented").struct.field("fr1"),
            pl.col("segmented").struct.field("fr2"),
            pl.col("segmented").struct.field("fr3"),
            pl.col("segmented").struct.field("fr4"),
            pl.col("segmented").struct.field("cdr1"),
            pl.col("segmented").struct.field("cdr2"),
            pl.col("segmented").struct.field("cdr3"),
        )
        .drop("segmented")
        .filter(
            pl.col("fr1").is_not_null()
            & pl.col("fr4").is_not_null()
            & (pl.col("fr1").str.len_chars() > 0)
            & (pl.col("fr4").str.len_chars() > 0)
        )
        .unique(subset=["fr1", "fr2", "fr3", "fr4"])
        .collect()
    )


# ---------------------------------------------------------------------------
# Nanobody input preparation (for Ray path)
# ---------------------------------------------------------------------------


def _join_numbered(seq: list) -> str:
    return "".join([x[1] for x in seq])


def prepare_nanobody(
    sequence: str,
) -> tuple[dict[str, list], dict[str, str], np.ndarray]:
    """Number a single nanobody sequence and produce the encoding array."""
    vh = number_single_sequence(sequence, "H", allowed_species=None)
    numbered = {"H": vh, "L": []}
    seq_dict = {"H": _join_numbered(vh), "L": ""}
    enc = get_encoding(seq_dict, "H")
    return numbered, seq_dict, enc


# ---------------------------------------------------------------------------
# Ray GPU worker
# ---------------------------------------------------------------------------


class NanobodyModelWorker:
    """Ray actor that holds nanobody models on a specific GPU."""

    def __init__(self, model_refs: list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import ray

        self.models = [ray.get(ref) for ref in model_refs]
        for m in self.models:
            m.to(self.device)
            m.eval()

    def predict(
        self, encoding: torch.Tensor, full_seq: str
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        encoding = encoding.to(self.device)
        with torch.no_grad():
            return [
                (atoms.detach().cpu(), feats.detach().cpu())
                for atoms, feats in (m(encoding, full_seq) for m in self.models)
            ]


class NanoBodyBuilder2Ray:
    """Ray-parallelized nanobody predictor for GPU inference."""

    def __init__(
        self,
        model_ids: tuple[int, ...] = (1, 2, 3, 4),
        weights_dir: str | Path | None = None,
    ):
        import ray

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if weights_dir is None:
            weights_dir = (
                Path(__file__).resolve().parent / "ImmuneBuilder" / "trained_model"
            )
        weights_dir = Path(weights_dir)

        self.model_refs: list = []
        for mid in model_ids:
            model_file = f"nanobody_model_{mid}"
            model = StructureModule(rel_pos_dim=64, embed_dim=embed_dim[model_file]).to(
                device
            )
            weights_path = weights_dir / model_file

            if not are_weights_ready(str(weights_path)):
                logger.info(f"Downloading weights for {model_file}...")
                download_file(model_urls[model_file], str(weights_path))

            model.load_state_dict(
                torch.load(str(weights_path), map_location=torch.device(device))
            )
            model.to(torch.get_default_dtype())
            model.eval()
            self.model_refs.append(ray.put(model))

    def create_workers(
        self,
        num_workers: int,
        cpu_per_worker: int = 1,
        gpu_frac_per_worker: float = 0.25,
    ) -> list:
        import ray

        remote_cls = ray.remote(NanobodyModelWorker).options(
            num_cpus=cpu_per_worker, num_gpus=gpu_frac_per_worker
        )
        return [remote_cls.remote(self.model_refs) for _ in range(num_workers)]

    def predict_batch(
        self,
        sequences: list[str],
        num_workers: int = 4,
        cpu_per_worker: int = 1,
        gpu_frac_per_worker: float = 0.25,
    ) -> abc.Iterator[tuple[str, Nanobody]]:
        """Yield (sequence, Nanobody) pairs using Ray workers."""
        import ray

        workers = self.create_workers(num_workers, cpu_per_worker, gpu_frac_per_worker)
        worker_cycle = cycle(workers)

        futures: list[tuple[object, str, dict]] = []
        for seq in sequences:
            try:
                numbered, seq_dict, enc = prepare_nanobody(seq)
            except Exception as e:
                logger.error(f"Failed to prepare {seq[:20]}...: {e}")
                continue

            encoding = torch.tensor(enc, dtype=torch.get_default_dtype())
            full_seq = seq_dict["H"]
            worker = next(worker_cycle)
            future = worker.predict.remote(encoding, full_seq)
            futures.append((future, seq, numbered))

        for future, seq, numbered in futures:
            try:
                output = ray.get(future)
                yield seq, Nanobody(numbered, output)
            except Exception as e:
                logger.error(f"Failed to predict {seq[:20]}...: {e}")


# ---------------------------------------------------------------------------
# Structure generation
# ---------------------------------------------------------------------------


def _save_nanobody_gz(
    nanobody: Nanobody, out_path: str, tmp_pdb: str, refinement: bool
) -> None:
    """Save a Nanobody prediction as a gzipped PDB."""
    if refinement:
        nanobody.save(tmp_pdb)
    else:
        nanobody.save_single_unrefined(tmp_pdb, index=nanobody.ranking[0])

    with open(tmp_pdb, "rb") as f_in:
        with gzip.open(out_path, "wb") as f_out:
            f_out.write(f_in.read())
    os.remove(tmp_pdb)


def generate_structures_gpu(
    sequences: list[str],
    output_dir: str,
    refinement: bool = True,
    num_workers: int = 4,
    gpu_frac_per_worker: float = 0.25,
) -> None:
    """Generate structures in parallel using Ray + GPU."""
    import ray

    os.makedirs(output_dir, exist_ok=True)

    # Filter to sequences not yet generated
    pending = [
        s
        for s in sequences
        if not os.path.exists(os.path.join(output_dir, f"{s}.pdb.gz"))
    ]
    if not pending:
        logger.info("All structures already generated.")
        return

    logger.info(
        f"Generating {len(pending)} structures using Ray ({num_workers} workers)..."
    )
    ray.init(ignore_reinit_error=True)

    predictor = NanoBodyBuilder2Ray()
    results = predictor.predict_batch(
        pending,
        num_workers=num_workers,
        gpu_frac_per_worker=gpu_frac_per_worker,
    )
    for seq, nanobody in tqdm(
        results, total=len(pending), desc="Generating structures", unit="seq"
    ):
        out_path = os.path.join(output_dir, f"{seq}.pdb.gz")
        tmp_pdb = os.path.join(output_dir, f"{seq}.pdb")
        try:
            _save_nanobody_gz(nanobody, out_path, tmp_pdb, refinement)
        except Exception as e:
            logger.error(f"Failed to save structure for {seq[:20]}...: {e}")

    ray.shutdown()


def generate_structures_cpu(
    sequences: list[str],
    output_dir: str,
    refinement: bool = True,
) -> None:
    """Generate structures sequentially on CPU."""
    from ImmuneBuilder.NanoBodyBuilder2 import NanoBodyBuilder2

    os.makedirs(output_dir, exist_ok=True)
    predictor = NanoBodyBuilder2()

    for seq in tqdm(sequences, desc="Generating structures", unit="seq"):
        out_path = os.path.join(output_dir, f"{seq}.pdb.gz")
        if os.path.exists(out_path):
            continue
        try:
            nanobody = predictor.predict({"H": seq})
            tmp_pdb = os.path.join(output_dir, f"{seq}.pdb")
            _save_nanobody_gz(nanobody, out_path, tmp_pdb, refinement)
        except Exception as e:
            logger.error(f"Failed to generate structure for {seq}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_parquet: str = typer.Argument(
        help="Path to input parquet file with nanobody sequences.",
    ),
    n_sample: int = typer.Option(10_000, help="Number of sequences to sample."),
    seed: int = typer.Option(42, help="Random seed for sampling."),
    no_refine: bool = typer.Option(
        False, help="Skip OpenMM refinement (faster but lower quality)."
    ),
    sequence_col: str = typer.Option(
        "Receptor Amino Acids", help="Column name containing sequences."
    ),
    num_workers: int = typer.Option(
        4, help="Number of Ray GPU workers (only used when CUDA is available)."
    ),
    gpu_frac: float = typer.Option(
        0.25, help="Fraction of GPU per Ray worker (only used when CUDA is available)."
    ),
) -> None:
    """Generate nanobody structures from a parquet dataset."""

    # 1. Segment all sequences and deduplicate on frameworks (lazy pipeline)
    logger.info(f"Segmenting and deduplicating sequences from {input_parquet}...")
    df = segment_and_deduplicate(input_parquet, sequence_col)
    logger.info(f"  {len(df):,} unique sequences after segmentation and dedup")

    # 2. Sample n_sample sequences
    if len(df) > n_sample:
        df = df.sample(n=n_sample, seed=seed)
    logger.info(f"Sampled {len(df):,} sequences")

    # 3. Save sampled sequences to parquet
    output_parquet = "filtered.parquet"
    df.write_parquet(output_parquet)
    logger.info(f"Saved to {output_parquet}")

    # 4. Generate structures
    output_dir = "generated"
    sequences = df[sequence_col].to_list()

    if CUDA_AVAILABLE:
        logger.info(f"CUDA available — using Ray with {num_workers} GPU workers")
        generate_structures_gpu(
            sequences,
            output_dir,
            refinement=not no_refine,
            num_workers=num_workers,
            gpu_frac_per_worker=gpu_frac,
        )
    else:
        logger.info("CUDA not available — running sequentially on CPU")
        generate_structures_cpu(sequences, output_dir, refinement=not no_refine)

    logger.info("Done!")


if __name__ == "__main__":
    app()
