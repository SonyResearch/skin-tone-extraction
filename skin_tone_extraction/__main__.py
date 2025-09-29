"""skin_tone extraction CLI."""

import os
import sys

import click
from rich.console import Console
from rich.table import Table

from skin_tone_extraction.dataset import get_dataset_config

from .batch_extract import batch_extract_df
from .extract import DEFAULT_METHOD, METHODS
from .image_collection import MaskedImageCollection

console = Console()


def _check_methods_all(methods):
    # Special case: Use ALL methods
    if methods == "all":
        methods = tuple(METHODS.keys())

    return methods


def _cli_extract(
    collection,
    methods,
    output_csv,
    limit=None,
):
    """Internal CLI extraction function.

    This function handles the core extraction logic, supporting single or multiple
    extraction methods. For multiple methods, it recursively calls itself for each
    method to generate separate output files.

    Args:
        collection: MaskedImageCollection instance containing images and masks
        methods: Extraction method name(s) - string, list, or tuple of method names
        output_csv: Output CSV file path(s) - single path or comma-separated paths
        limit: Optional limit on number of images to process

    Raises:
        SystemExit: If validation fails, methods don't exist, or file operations fail
    """
    # Check if method is all
    methods = _check_methods_all(methods)

    # Normalize method to a tuple
    if isinstance(methods, list):
        methods = tuple(methods)

    # Support for multiple methods
    if len(methods) > 1:
        # Immediately validate that all methods exist
        for m in methods:
            if m not in METHODS:
                console.print(f"[red]Unknown method: {m}[/red]")
                console.print(
                    "[red]Available methods: {}[/red]".format(", ".join(METHODS.keys()))
                )
                sys.exit(1)

        # Allow output_csv to be a comma-separated string
        output_csvs = None
        if output_csv is not None:
            output_csvs = [c.strip() for c in str(output_csv).split(",") if c.strip()]
            if len(output_csvs) == 1 and len(methods) > 1:
                console.print(
                    "[red]Cannot specify a single --output-csv when using multiple "
                    "methods.[/red]"
                )
                sys.exit(1)
            if len(output_csvs) > 1 and len(output_csvs) != len(methods):
                console.print(
                    "[red]Number of output_csv files must match number of methods."
                    "[/red]"
                )
                sys.exit(1)
        console.print(
            "[yellow]Running extraction for multiple methods: {}[/yellow]".format(
                ", ".join(methods)
            )
        )
        for idx, m in enumerate(methods):
            ocsv = None
            if output_csvs is not None:
                ocsv = output_csvs[idx]

            _cli_extract(
                collection=collection,
                methods=m,
                output_csv=ocsv,
                limit=limit,
            )
        console.print("[green]All methods completed successfully![/green]")
        return

    # Single method
    method = methods[0]

    # Determine default output path
    if output_csv is None:
        img_paths = collection.image_paths
        images_base = os.path.dirname(img_paths[0])
        safe_images_base = images_base.strip(os.sep).replace(os.sep, "_")
        output_csv = f"data/extracted/skin_tone-{safe_images_base}-{method}.csv"

    # Optionally limit the number of images (for testing)
    working_collection = collection
    parsed_image_ids = collection.image_ids
    if limit is not None:
        working_collection = collection.slice(0, limit)
        if parsed_image_ids:
            parsed_image_ids = parsed_image_ids[:limit]

    # Display configuration/settings as a rich table
    config_table = Table(title="Extraction Configuration")
    config_table.add_column("Setting", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="magenta")
    config_table.add_row("Method", str(method))
    config_table.add_row("No of Images", str(len(working_collection)))
    config_table.add_row("Mask Value", str(working_collection.mask_value))
    config_table.add_row("Output CSV", str(output_csv))
    console.print(config_table)

    results_df = batch_extract_df(collection=working_collection, method=method)

    # Add image_id column if available
    if parsed_image_ids is not None and len(parsed_image_ids) == len(results_df):
        results_df.insert(0, "image_id", parsed_image_ids)

    # Write output file
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    console.print(f"[green]Results saved to {output_csv}[/green]")


@click.group()
def cli():
    """skin_tone extraction CLI."""


@cli.command("extract-dataset")
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="Dataset preset to use.",
)
@click.option(
    "--method",
    default=[DEFAULT_METHOD],
    type=click.Choice(list(METHODS.keys())),
    multiple=True,
    help="Extraction method (optional, multiple possible).",
)
@click.option(
    "--output-csv",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, writable=True),
    help="Output CSV file path.",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    show_default=True,
    help="Process only the first X images (optional).",
)
def extract_dataset(dataset, method, output_csv, limit):
    """Extract skin_tone values from a predefined dataset."""
    dataset_config = get_dataset_config(dataset)
    collection = MaskedImageCollection.from_dataset_config(dataset_config)

    # Check if method is all (this is required here to handle csv file generation)
    methods = _check_methods_all(method)

    # Generate default output_csv path
    if output_csv is None:
        # Generate one output_csv per method
        output_csvs = [f"data/extracted/skin_tone-{dataset}-{m}.csv" for m in methods]
        os.makedirs(os.path.dirname(output_csvs[0]), exist_ok=True)

        # Join all output CSVs into a single string
        output_csv = ",".join(output_csvs)

    # Run extraction
    _cli_extract(
        collection=collection,
        methods=methods,
        output_csv=output_csv,
        limit=limit,
    )


@cli.command("extract")
@click.option(
    "--images",
    type=str,
    required=True,
    help="Directory with input images or a pattern with {image_id}, "
    "e.g. path/to/imgs/{image_id}.png",
)
@click.option(
    "--masks",
    type=str,
    required=True,
    help="Directory with mask images or a pattern with {image_id}, "
    "e.g. path/to/masks/{image_id}.png",
)
@click.option(
    "--mask-value",
    required=True,
    type=int,
    help="Mask value for skin pixels.",
)
@click.option(
    "--method",
    default=[DEFAULT_METHOD],
    type=click.Choice(list(METHODS.keys())),
    multiple=True,
    help="Extraction method (optional, multiple possible).",
)
@click.option(
    "--output-csv",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, writable=True),
    help="Output CSV file path.",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    show_default=True,
    help="Process only the first X images (optional).",
)
def extract(images, masks, mask_value, method, output_csv, limit):
    """Extract skin_tone values from images and masks provided manually."""
    # If multiple methods, output_csv must not be specified
    if "," in method and output_csv is not None:
        console.print(
            "[red]Cannot specify --output-csv when using multiple methods.[/red]"
        )
        sys.exit(1)

    collection = MaskedImageCollection(
        images=images,
        masks=masks,
        mask_value=mask_value,
    )

    _cli_extract(
        collection=collection,
        methods=method,
        output_csv=output_csv,
        limit=limit,
    )


if __name__ == "__main__":
    cli()
