"""Tests for image_collection functions."""

import os
from unittest.mock import Mock, patch

import pytest

from res_code.skin_tone.image_collection import MaskedImageCollection


def test_init_with_lists():
    """Test initialization with list inputs."""
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    mask_paths = ["mask1.jpg", "mask2.jpg", "mask3.jpg"]
    image_ids = ["id1", "id2", "id3"]

    collection = MaskedImageCollection(
        images=image_paths, masks=mask_paths, mask_value=1, image_ids=image_ids
    )

    assert len(collection) == 3
    assert collection.mask_value == 1
    assert collection.image_paths == image_paths
    assert collection.mask_paths == mask_paths
    assert collection.image_ids == image_ids


def test_init_with_lists_no_image_ids():
    """Test initialization with list inputs but no image IDs."""
    image_paths = ["img1.jpg", "img2.jpg"]
    mask_paths = ["mask1.jpg", "mask2.jpg"]

    collection = MaskedImageCollection(
        images=image_paths, masks=mask_paths, mask_value=2
    )

    assert len(collection) == 2
    assert collection.mask_value == 2
    assert collection.image_ids is None


def test_init_with_mismatched_list_lengths():
    """Test that mismatched list lengths raise ValueError."""
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    mask_paths = ["mask1.jpg", "mask2.jpg"]  # One less

    with pytest.raises(ValueError, match="Length of images and masks lists must match"):
        MaskedImageCollection(images=image_paths, masks=mask_paths, mask_value=1)


def test_init_with_mismatched_image_ids_length():
    """Test that mismatched image_ids length raises ValueError."""
    image_paths = ["img1.jpg", "img2.jpg"]
    mask_paths = ["mask1.jpg", "mask2.jpg"]
    image_ids = ["id1"]  # One less

    with pytest.raises(ValueError, match="Length of image_ids must match"):
        MaskedImageCollection(
            images=image_paths, masks=mask_paths, mask_value=1, image_ids=image_ids
        )


def test_get_id():
    """Test getting image ID by index."""
    collection = MaskedImageCollection(
        images=["img1.jpg", "img2.jpg"],
        masks=["mask1.jpg", "mask2.jpg"],
        mask_value=1,
        image_ids=["id1", "id2"],
    )

    assert collection.get_id(0) == "id1"
    assert collection.get_id(1) == "id2"


def test_get_id_no_image_ids():
    """Test getting image ID when none are provided."""
    collection = MaskedImageCollection(
        images=["img1.jpg"], masks=["mask1.jpg"], mask_value=1
    )

    assert collection.get_id(0) is None


def test_get_id_out_of_range():
    """Test that get_id raises IndexError for invalid indices."""
    collection = MaskedImageCollection(
        images=["img1.jpg"], masks=["mask1.jpg"], mask_value=1, image_ids=["id1"]
    )

    with pytest.raises(IndexError, match="Index out of range"):
        collection.get_id(1)


def test_slice():
    """Test slicing functionality."""
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    mask_paths = ["mask1.jpg", "mask2.jpg", "mask3.jpg", "mask4.jpg"]
    image_ids = ["id1", "id2", "id3", "id4"]

    collection = MaskedImageCollection(
        images=image_paths, masks=mask_paths, mask_value=1, image_ids=image_ids
    )

    # Test basic slice
    sliced = collection.slice(1, 3)
    assert len(sliced) == 2
    assert sliced.image_paths == ["img2.jpg", "img3.jpg"]
    assert sliced.mask_paths == ["mask2.jpg", "mask3.jpg"]
    assert sliced.image_ids == ["id2", "id3"]
    assert sliced.mask_value == 1

    # Test slice with step
    sliced_step = collection.slice(0, None, 2)
    assert len(sliced_step) == 2
    assert sliced_step.image_paths == ["img1.jpg", "img3.jpg"]

    # Test slice without image_ids
    collection_no_ids = MaskedImageCollection(
        images=image_paths, masks=mask_paths, mask_value=1
    )
    sliced_no_ids = collection_no_ids.slice(0, 2)
    assert sliced_no_ids.image_ids is None


@patch("res_code.skin_tone.image_collection.MaskedImage")
def test_getitem_creates_masked_image(mock_masked_image):
    """Test that __getitem__ creates MaskedImage instances when needed."""
    mock_instance = Mock()
    mock_masked_image.return_value = mock_instance

    collection = MaskedImageCollection(
        images=["img1.jpg", "img2.jpg"],
        masks=["mask1.jpg", "mask2.jpg"],
        mask_value=1,
        image_ids=["id1", "id2"],
    )

    # First access should create the instance
    result = collection[0]
    assert result == mock_instance
    mock_masked_image.assert_called_once_with(
        img_path="img1.jpg",
        mask_path="mask1.jpg",
        mask_value=1,
        id="id1",
        collection=collection,
    )

    # Second access should return cached instance
    mock_masked_image.reset_mock()
    result2 = collection[0]
    assert result2 == mock_instance
    mock_masked_image.assert_not_called()  # Should not create a new instance


def test_getitem_out_of_range():
    """Test that __getitem__ raises IndexError for invalid indices."""
    collection = MaskedImageCollection(
        images=["img1.jpg"], masks=["mask1.jpg"], mask_value=1
    )

    with pytest.raises(IndexError, match="Index out of range"):
        collection[1]

    with pytest.raises(IndexError, match="Index out of range"):
        collection[-1]


@patch("res_code.skin_tone.image_collection.MaskedImage")
def test_iteration(mock_masked_image):
    """Test iteration over the collection."""
    mock_instances = [Mock() for _ in range(3)]
    mock_masked_image.side_effect = mock_instances

    collection = MaskedImageCollection(
        images=["img1.jpg", "img2.jpg", "img3.jpg"],
        masks=["mask1.jpg", "mask2.jpg", "mask3.jpg"],
        mask_value=1,
    )

    results = list(collection)
    assert len(results) == 3
    assert results == mock_instances


def test_clear_cache():
    """Test clearing the cache of loaded images."""
    collection = MaskedImageCollection(
        images=["img1.jpg", "img2.jpg"], masks=["mask1.jpg", "mask2.jpg"], mask_value=1
    )

    # Access an item to populate cache
    with patch("res_code.skin_tone.image_collection.MaskedImage") as mock:
        mock.return_value = Mock()
        _ = collection[0]
        assert len(collection._loaded_images) == 1

    # Clear cache
    collection.clear_cache()
    assert len(collection._loaded_images) == 0


def test_directory_parsing_with_files(tmp_path):
    """Test directory parsing with actual files."""
    # Create test directories and files
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    # Create matching files in both directories
    filenames = ["file1.png", "file2.png", "file3.png"]
    for filename in filenames:
        (img_dir / filename).write_text("dummy image")
        (mask_dir / filename).write_text("dummy mask")

    collection = MaskedImageCollection(
        images=str(img_dir), masks=str(mask_dir), mask_value=2
    )

    assert len(collection) == 3
    assert collection.image_ids is None  # No IDs in directory mode

    # Verify all files are included
    img_basenames = [os.path.basename(p) for p in collection.image_paths]
    mask_basenames = [os.path.basename(p) for p in collection.mask_paths]
    assert sorted(img_basenames) == sorted(filenames)
    assert sorted(mask_basenames) == sorted(filenames)


def test_pattern_parsing_no_files():
    """Test pattern parsing when no files match."""
    with pytest.raises(ValueError, match="No images found for pattern"):
        MaskedImageCollection(
            images="/nonexistent/path/{image_id}.jpg",
            masks="/nonexistent/path/{image_id}_mask.jpg",
            mask_value=1,
        )


def test_directory_parsing_no_common_files(tmp_path):
    """Test directory parsing when no common files exist."""
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    # Create different files in each directory
    (img_dir / "img1.jpg").write_text("dummy")
    (mask_dir / "mask2.jpg").write_text("dummy")

    with pytest.raises(ValueError, match="No matching files found in both directories"):
        MaskedImageCollection(images=str(img_dir), masks=str(mask_dir), mask_value=1)


def test_tuple_inputs_parsing(tmp_path):
    """Test parsing with tuple inputs."""
    # Create test files in multiple directories
    dir1 = tmp_path / "set1"
    dir2 = tmp_path / "set2"
    dir1.mkdir()
    dir2.mkdir()

    # First set of files
    (dir1 / "img1.jpg").write_text("dummy")
    (dir1 / "mask1.jpg").write_text("dummy")

    # Second set of files
    (dir2 / "img2.jpg").write_text("dummy")
    (dir2 / "mask2.jpg").write_text("dummy")

    collection = MaskedImageCollection(
        images=(str(dir1) + "/img{image_id}.jpg", str(dir2) + "/img{image_id}.jpg"),
        masks=(str(dir1) + "/mask{image_id}.jpg", str(dir2) + "/mask{image_id}.jpg"),
        mask_value=1,
    )

    assert len(collection) == 2
    # Should include files from both directories
    img_names = [os.path.basename(p) for p in collection.image_paths]
    assert "img1.jpg" in img_names
    assert "img2.jpg" in img_names
