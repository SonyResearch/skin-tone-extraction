"""Tests for CLI related functions."""

import os

import pytest
from click.testing import CliRunner

from res_code.skin_tone.__main__ import (
    cli,
)
from res_code.skin_tone.dataset import DatasetConfig, get_dataset_config
from res_code.skin_tone.image_collection import MaskedImageCollection


def create_dummy_files(dir_path, names):
    os.makedirs(dir_path, exist_ok=True)
    for name in names:
        with open(os.path.join(dir_path, name), "w") as f:
            f.write("dummy")


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_cli_pattern_mode(tmp_path):
    # Create dummy files for pattern mode
    img_dir = tmp_path / "imgs"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    ids = ["x"]
    for i in ids:
        (img_dir / f"{i}.png").write_text("img")
        (mask_dir / f"{i}_mask.png").write_text("mask")
    images_pattern = str(img_dir / "{image_id}.png")
    masks_pattern = str(mask_dir / "{image_id}_mask.png")
    runner = CliRunner()
    # Patch batch_extract_df to avoid actual processing
    import res_code.skin_tone.__main__ as main_mod

    class DummyDF:
        def insert(self, *a, **kw):
            pass

        def to_csv(self, path, index):
            open(path, "w").write("dummy")

        def __len__(self):
            return 1

    main_mod.batch_extract_df = lambda *a, **kw: DummyDF()
    result = runner.invoke(
        cli,
        [
            "extract",
            "--images",
            images_pattern,
            "--masks",
            masks_pattern,
            "--mask-value",
            "1",
            "--output-csv",
            str(tmp_path / "out.csv"),
        ],
    )
    assert result.exit_code == 0
    assert "Results saved" in result.output
    assert os.path.exists(tmp_path / "out.csv")


def test_get_dataset_config_success(tmp_path, monkeypatch):
    # Create a dummy datasets.py file
    datasets_py = tmp_path / "datasets.py"
    datasets_py.write_text(
        "class DummyConfig:\n"
        "    images = ['img1.png']\n"
        "    masks = ['mask1.png']\n"
        "    mask_value = 1\n"
        "    image_ids = ['id1']\n"
        "dummy = DummyConfig\n"
    )

    # Patch DatasetConfig to be DummyConfig's base
    class DummyBase(DatasetConfig):
        pass

    monkeypatch.setattr("res_code.skin_tone.dataset.DatasetConfig", DummyBase)
    # Should load and instantiate DummyConfig
    config = get_dataset_config("dummy", str(datasets_py))
    assert hasattr(config, "images")
    assert config.images == ["img1.png"]
    assert config.mask_value == 1


def test_get_dataset_config_missing_file(tmp_path):
    missing_path = tmp_path / "notfound.py"
    with pytest.raises(FileNotFoundError):
        get_dataset_config("dummy", str(missing_path))


def test_get_dataset_config_missing_dataset(tmp_path):
    datasets_py = tmp_path / "datasets.py"
    datasets_py.write_text("foo = 123\n")
    with pytest.raises(ValueError):
        get_dataset_config("bar", str(datasets_py))


def test_cli_extract_invalid_method(tmp_path):
    """Test _cli_extract with invalid method names."""
    # Create dummy image and mask files
    img = tmp_path / "img.png"
    mask = tmp_path / "mask.png"
    img.write_text("img")
    mask.write_text("mask")

    # Create a MaskedImageCollection
    collection = MaskedImageCollection(
        images=[str(img)], masks=[str(mask)], mask_value=1
    )

    # Patch batch_extract_df to avoid actual processing
    import res_code.skin_tone.__main__ as main_mod

    main_mod.batch_extract_df = lambda *a, **kw: []

    # Test with invalid methods
    with pytest.raises(SystemExit):
        main_mod._cli_extract(
            collection=collection,
            methods=["merler", "not_a_method"],
            output_csv="out1.csv,out2.csv",
        )


def test_cli_extract_single_method(tmp_path):
    """Test _cli_extract with a single valid method."""
    # Create dummy image and mask files
    img = tmp_path / "img.png"
    mask = tmp_path / "mask.png"
    img.write_text("img")
    mask.write_text("mask")

    # Create a MaskedImageCollection
    collection = MaskedImageCollection(
        images=[str(img)], masks=[str(mask)], mask_value=1, image_ids=["test_id"]
    )

    # Mock batch_extract_df
    import res_code.skin_tone.__main__ as main_mod

    class DummyDF:
        def __init__(self):
            self.data = []

        def insert(self, *a, **kw):
            pass

        def to_csv(self, path, index):
            with open(path, "w") as f:
                f.write("dummy_csv_content")

        def __len__(self):
            return 1

    main_mod.batch_extract_df = lambda *a, **kw: DummyDF()

    # Test with valid method
    output_csv = str(tmp_path / "output.csv")
    main_mod._cli_extract(
        collection=collection,
        methods=["average"],
        output_csv=output_csv,
    )

    # Check output file was created
    assert os.path.exists(output_csv)


def test_cli_extract_with_limit(tmp_path):
    """Test _cli_extract with a limit on number of images."""
    # Create multiple dummy image and mask files
    images = []
    masks = []
    for i in range(5):
        img = tmp_path / f"img{i}.png"
        mask = tmp_path / f"mask{i}.png"
        img.write_text(f"img{i}")
        mask.write_text(f"mask{i}")
        images.append(str(img))
        masks.append(str(mask))

    # Create a MaskedImageCollection
    collection = MaskedImageCollection(images=images, masks=masks, mask_value=1)

    # Mock batch_extract_df to track how many images are processed
    import res_code.skin_tone.__main__ as main_mod

    processed_count = 0

    def mock_batch_extract_df(collection, **kwargs):
        nonlocal processed_count
        processed_count = len(collection)

        class DummyDF:
            def insert(self, *a, **kw):
                pass

            def to_csv(self, path, index):
                with open(path, "w") as f:
                    f.write("dummy_csv_content")

            def __len__(self):
                return processed_count

        return DummyDF()

    main_mod.batch_extract_df = mock_batch_extract_df

    # Test with limit
    output_csv = str(tmp_path / "output_limited.csv")
    main_mod._cli_extract(
        collection=collection, methods=["average"], output_csv=output_csv, limit=3
    )

    # Check that only 3 images were processed
    assert processed_count == 3
    assert os.path.exists(output_csv)
