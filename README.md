# skin_tone_extraction

`skin_tone_extraction` is a Python package for automatic skin tone extraction from images using various different approaches. The package provides both a programmatic API and command-line interface for extracting skin tone from facial images with skin masks.

## Installation

Install this library using `pip`:
```bash
pip install git+https://github.com/SonyResearch/skin-tone-extraction.git
```

## Usage

The easiest way to use the package is via its CLI. Point it to a list of images and masks and it will extract skin tone information for each image.

```bash
# Extract from directories
python -m skin_tone_extraction extract \
  --images /path/to/images
  --masks /path/to/masks \
  --mask-value 255 # Which value in the mask corresponds to facial skin?

# You can also use {image_id} as a placeholder for more complex scenarios
python -m skin_tone_extraction extract \
  --images "/data/images/{image_id}.jpg" \
  --masks "/data/images/mask_{image_id}.png" \
  --mask-value 1

# If you want to better understand how a method works, turn on debugging mode
python -m skin_tone_extraction extract \
  --images /path/to/images \
  --masks /path/to/masks \
  --mask-value 255 \
  --debug # Learn more about how a method works
```

The extracted data will be automatically saved as a CSV-file.

### Methods

The package supports several different methods for extraction:

- `average`: Overall average color of skin pixels (default)
- `mode`: Most frequent color of skin pixels
- `thong`: Clustering-based approach from Thong et al. (2023)
- `krishnapriya`: Drawing from Krishnapriya & Bourlai (2016)
- `merler`: Drawing from Merler et al. (2019)
- `all`: Extract all methods

```bash
# You can also extract multiple methods at once
python -m skin_tone_extraction extract \
  --images /path/to/images \
  --masks /path/to/masks \
  --mask-value 255 \
  --method thong \
  --method average
```

### Output Format

All extraction methods return the following metrics (either in a CSV file or in individual dictionaries):

- **Skin tone metrics**: 
  - `lum`: Luminance in CIELAB space
  - `hue`: Hue angle in degrees
  - `ita`: Individual Typology Angle in degrees
  - `cita`: Corrected ITA using absolute b* values
- **Color channels**: `red`, `green`, `blue` (0-1 range)
- **LAB color space**: `lum` (luminance), `lab_a`, `lab_b` 

Please note that we do not recommend usage of ITA (even in corrected form), as it does not represent diverse skin tone well. We refer to the paper for a more detailed discussion of the issues with ITA.

### Python API

#### Basic usage

```python
from skin_tone_extraction import extract_skin_tone_from_paths

# Extract skin tone from single image
result = extract_skin_tone_from_paths(
    img_path="face.jpg",
    mask_path="face_mask.png", 
    mask_value=255,
    method="average"
)

print(result)
# {'lum': 60.48, 'hue': 50.55, 'red': 0.78, 'green': 0.65, 'blue': 0.52, 'lab_a': 8.1, ...}
```

#### Batch processing with collections

```python
from skin_tone_extraction.image_collection import MaskedImageCollection
from skin_tone_extraction.batch_extract import batch_extract_df

# Create collection from directories
collection = MaskedImageCollection(
    images="/path/to/images",
    masks="/path/to/masks", 
    mask_value=255
)

# Or from patterns
collection = MaskedImageCollection(
    images="/data/images/{image_id}.jpg",
    masks="/data/masks/{image_id}.png",
    mask_value=1
)

# Extract skin tones for all images
results_df = batch_extract_df(collection=collection, method="average")
print(results_df.head())
```

## Dataset Configuration

The package supports predefined dataset configurations for easier reuse and sharing of dataset definitions or to specify more complex scenarios, where e.g. a dataset needs to be loaded to extract image paths.

### Defining datasets

Create a `datasets.py` file in your working directory to define reusable dataset configurations:

```python
from skin_tone_extraction import DatasetConfig

class MyDataset(DatasetConfig):
    """My custom facial image dataset."""
    
    # Use file patterns with {image_id} placeholder  
    images = "/data/faces/{image_id}.jpg"
    masks = "/data/face_masks/{image_id}.png"
    mask_value = 255

class ComplexDataset(DatasetConfig):
    """Dataset requiring complex path resolution."""
    
    mask_value = 1
    
    def __init__(self):
        # Dynamic path generation from CSV, database, etc.
        import pandas as pd
        df = pd.read_csv("/data/dataset_manifest.csv")
        
        self.images = df['image_path'].tolist()
        self.masks = df['mask_path'].tolist() 
        self.image_ids = df['id'].tolist()
```

#### Extract from predefined datasets

```bash
# Extract using a dataset configuration
python -m skin_tone_extraction extract-dataset --dataset MyDataset

# Use multiple methods
python -m skin_tone_extraction extract-dataset \
    --dataset MyDataset \
    --method average \
    --method krishnapriya

# Limit processing to first 100 images (useful when trying out extraction)
python -m skin_tone_extraction extract-dataset \
    --dataset ComplexDataset \
    --method average \
    --limit 100
```

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:

```bash
cd PACKAGENAME
python -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
python -m pip install -e '.[test]'
```

To run the tests:

```bash
python -m pytest
```

### Formatting

Ruff is used for linting and black for formatting. Formatting can be automatically checked / applied wherever possible via `black . && ruff check . --fix && ruff format .`.

## Acknowledgements

This package builts on template code by @jeroneandrews-sony, @athoag-sony and @jinruxue-sony.

Some of the extraction methods in this package build on prior work, please cite these works appropriately if you use one of the methods. The documentation of a particular extraction class will reference its source / inspiration.

Krishnapriya, K. S., Pangelinan, G., King, M. C., & Bowyer, K. W. (2022). Analysis of Manual and Automated Skin Tone Assignments. 429–438. https://openaccess.thecvf.com/content/WACV2022W/DVPB/html/Krishnapriya_Analysis_of_Manual_and_Automated_Skin_Tone_Assignments_WACVW_2022_paper.html
Merler, M., Ratha, N., Feris, R. S., & Smith, J. R. (2019). Diversity in Faces (No. arXiv:1901.10436). arXiv. https://doi.org/10.48550/arXiv.1901.10436
Thong, W., Joniak, P., & Xiang, A. (2023). Beyond Skin Tone: A Multidimensional Measure of Apparent Skin Color. 4903–4913. https://openaccess.thecvf.com/content/ICCV2023/html/Thong_Beyond_Skin_Tone_A_Multidimensional_Measure_of_Apparent_Skin_Color_ICCV_2023_paper.html

## License

The code in this repositroy is licensed under the [MIT license](./LICENSE).
