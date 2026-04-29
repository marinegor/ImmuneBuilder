"""Generate nanobody structures from a parquet dataset.

Steps:
1. Number sequences using immunum (IMGT scheme)
2. Remove duplicates on (fr1, fr2, fr3, fr4)
3. Sample 10,000 random sequences and save to <timestamp>.parquet
4. Generate structures using NanoBodyBuilder2, save as <sequence>.pdb.gz
"""

import gzip
import os
from datetime import datetime

import polars as pl
import typer
from loguru import logger
from tqdm import tqdm

app = typer.Typer()


def segment_and_deduplicate(path: str, sequence_col: str) -> pl.DataFrame:
    """Segment all sequences and deduplicate on frameworks in a single lazy pipeline."""
    import immunum.polars as imp

    return (
        pl.scan_parquet(path)
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
        .filter(pl.col("fr1").is_not_null())
        .unique(subset=["fr1", "fr2", "fr3", "fr4"])
        .collect()
    )


def generate_structures(
    sequences: list[str],
    output_dir: str,
    refinement: bool = True,
) -> None:
    """Generate nanobody structures and save as gzipped PDB files."""
    from ImmuneBuilder.NanoBodyBuilder2 import NanoBodyBuilder2

    os.makedirs(output_dir, exist_ok=True)
    predictor = NanoBodyBuilder2()

    for seq in tqdm(sequences, desc="Generating structures", unit="seq"):
        out_path = os.path.join(output_dir, f"{seq}.pdb.gz")
        if os.path.exists(out_path):
            continue
        try:
            nanobody = predictor.predict({"H": seq})

            # Write unrefined PDB to a temp file, then gzip
            tmp_pdb = os.path.join(output_dir, f"{seq}.pdb")
            if refinement:
                nanobody.save(tmp_pdb)
            else:
                nanobody.save_single_unrefined(tmp_pdb, index=nanobody.ranking[0])

            with open(tmp_pdb, "rb") as f_in:
                with gzip.open(out_path, "wb") as f_out:
                    f_out.write(f_in.read())
            os.remove(tmp_pdb)

        except Exception as e:
            logger.error(f"Failed to generate structure for {seq}: {e}")


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
    output_dir = os.path.join("generated")
    logger.info(f"Generating structures in {output_dir}...")
    sequences = df[sequence_col].to_list()
    generate_structures(sequences, output_dir, refinement=not no_refine)

    logger.info("Done!")


if __name__ == "__main__":
    app()
