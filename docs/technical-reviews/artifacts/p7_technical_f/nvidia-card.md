---
license: cc-by-4.0
task_categories:
  - object-detection
  - image-to-text
language:
  - en
  - ja
  - ko
  - ru
  - zh
tags:
  - ocr
  - text-detection
  - text-recognition
  - synthetic-data
  - synthdog
  - hdf5
  - nvidia
  - nemotron
pretty_name: OCR Synthetic Multilingual v1
size_categories:
  - 10M<n<100M
---

# OCR-Synthetic-Multilingual-v1

## Dataset Description

Large-scale synthetically generated OCR training dataset for multilingual text detection and recognition. The data was produced using a heavily modified and extended version of [SynthDoG](https://github.com/clovaai/donut/tree/master/synthdog) (Synthetic Document Generator), originally introduced in the [Donut](https://github.com/clovaai/donut) project by Kim et al.

This dataset was used to train [**Nemotron OCR v2**](https://huggingface.co/nvidia/nemotron-ocr-v2), a state-of-the-art multilingual OCR model that is part of the [NVIDIA NeMo Retriever](https://developer.nvidia.com/nemo-retriever/) collection.

This dataset is ready for commercial/non-commercial use.

## Dataset Owner(s):
NVIDIA Corporation

## Dataset Creation Date:
April 15, 2026

## License/Terms of Use: 
Dataset Governing Terms: Use of the dataset is governed by the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Intended Usage:
This dataset is intended for machine learning researchers, AI engineers, and developers working on information retrieval with OCR.

## Dataset Characterization
** Data Collection Method<br>
* [Hybrid: Human, Automated, Synthetic]

** Labeling Method<br>
* [Not Applicable] <br>

## Dataset Format

```
OCR-Synthetic-Multilingual-v1/
├── en/
│   ├── train/
│   │   ├── train_000.h5
│   │   ├── train_001.h5
│   │   └── ...
│   ├── test/
│   │   └── ...
│   └── validation/
│       └── ...
├── ja/
│   └── ...
├── ko/
│   └── ...
├── ru/
│   └── ...
├── zh_hans/
│   └── ...
└── zh_hant/
    └── ...
```

## Format — HDF5

Each `.h5` file contains the following datasets (HDF5 terminology):

| Key           | Type                          | Description                                           |
|---------------|-------------------------------|-------------------------------------------------------|
| `images`      | object (variable-length bytes)| JPEG-encoded image bytes, one entry per sample         |
| `annotations` | object (variable-length str)  | JSON string per sample containing bounding-box annotations |
| `dimensions`  | int array `[H, W]`           | Original image dimensions                              |
| `labels`      | object (string)               | Full-page text label                                   |
| `qualities`   | int                           | JPEG quality used during encoding (typically 100)      |
| `sample_ids`  | int                           | Unique sample identifier                               |

## Annotation JSON Schema

Each entry in `annotations` is a JSON object:

```json
{
  "word_bboxes": [
    {
      "text": "example word or phrase",
      "bbox": [x, y, w, h],
      "quad": [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    }
  ],
  "line_bboxes": [
    {
      "text": "full line of text",
      "bbox": [x, y, w, h],
      "quad": [[x0,y0], [x1,y1], [x2,y2], [x3,y3]],
      "para_idx": 0,
      "line_idx": 0,
      "word_indices": [0, 1, 2]
    }
  ],
  "para_bboxes": [...],
  "relation_graph": [
    [[0], [1], [2]],
    [[3], [4]]
  ]
}
```

### Bounding Box Levels

- **`word_bboxes`** — One entry per word/phrase rendered as a single unit. Each contains `text`, an axis-aligned `bbox [x, y, w, h]`, and a 4-point `quad`.
- **`line_bboxes`** — One entry per text line. Includes all `word_bboxes` fields plus `para_idx` (paragraph index), `line_idx` (line index within the paragraph), and `word_indices` (indices into `word_bboxes` that compose this line).
- **`para_bboxes`** — One entry per paragraph bounding box.
- **`relation_graph`** — Nested list encoding reading order: `relation_graph[para][sentence]` gives a list of word/line indices belonging to that sentence within the paragraph.

### Quad Vertex Convention

Quads are 4-point polygons stored as `[[x0,y0], [x1,y1], [x2,y2], [x3,y3]]` in clockwise order:

```
v0 -------- v1
|            |
v3 -------- v2
```

## Loading Example

```python
import h5py, io, json
from PIL import Image

with h5py.File("en/train/train_000.h5", "r") as f:
    img_bytes = f["images"][0]
    image = Image.open(io.BytesIO(img_bytes.tobytes())).convert("RGB")

    annotation = json.loads(f["annotations"][0])
    for line in annotation["line_bboxes"]:
        print(line["text"], line["quad"])
```

---
## Dataset Quantification
# Languages

| Subfolder   | Language               | Total Samples | Train          | Test           | Validation     |
|-------------|------------------------|---------------|----------------|----------------|----------------|
| `en`        | English                | 1,825,089     | 1,460,304 (63) | 183,629 (63)   | 181,156 (63)   |
| `ja`        | Japanese               | 1,889,137     | 1,502,712 (67) | 193,779 (67)   | 192,646 (67)   |
| `ko`        | Korean                 | 2,269,540     | 1,814,994 (78) | 227,091 (78)   | 227,455 (78)   |
| `ru`        | Russian                | 1,724,733     | 1,380,404 (59) | 171,678 (59)   | 172,651 (59)   |
| `zh_hans`   | Chinese (Simplified)   | 2,335,343     | 1,914,948 (83) | 210,143 (73)   | 210,252 (73)   |
| `zh_hant`   | Chinese (Traditional)  | 2,214,304     | 1,772,280 (77) | 221,867 (77)   | 220,157 (77)   |
| **Total**   |                        | **12,258,146** | **9,845,642**  | **1,208,187**  | **1,204,317**  |

> Numbers in parentheses are the number of `.h5` files per split.

## Related Model

This dataset was created to train the detection, recognition, and relational components of [**Nemotron OCR v2**](https://huggingface.co/nvidia/nemotron-ocr-v2). See the model card for architecture details, evaluation results, and usage instructions.

## Per-Language Details

### English (`en`)

| Property       | Value                                |
|----------------|--------------------------------------|
| Language       | English (`en`)                       |
| Total Samples  | 1,825,089                            |
| Train          | 1,460,304 samples (63 files)         |
| Test           | 183,629 samples (63 files)           |
| Validation     | 181,156 samples (63 files)           |

### Japanese (`ja`)

| Property       | Value                                |
|----------------|--------------------------------------|
| Language       | Japanese (`ja`)                      |
| Total Samples  | 1,889,137                            |
| Train          | 1,502,712 samples (67 files)         |
| Test           | 193,779 samples (67 files)           |
| Validation     | 192,646 samples (67 files)           |

### Korean (`ko`)

| Property       | Value                                |
|----------------|--------------------------------------|
| Language       | Korean (`ko`)                        |
| Total Samples  | 2,269,540                            |
| Train          | 1,814,994 samples (78 files)         |
| Test           | 227,091 samples (78 files)           |
| Validation     | 227,455 samples (78 files)           |

### Russian (`ru`)

| Property       | Value                                |
|----------------|--------------------------------------|
| Language       | Russian (`ru`)                       |
| Total Samples  | 1,724,733                            |
| Train          | 1,380,404 samples (59 files)         |
| Test           | 171,678 samples (59 files)           |
| Validation     | 172,651 samples (59 files)           |

### Chinese Simplified (`zh_hans`)

| Property       | Value                                |
|----------------|--------------------------------------|
| Language       | Chinese (Simplified) (`zh_hans`)     |
| Total Samples  | 2,335,343                            |
| Train          | 1,914,948 samples (83 files)         |
| Test           | 210,143 samples (73 files)           |
| Validation     | 210,252 samples (73 files)           |

### Chinese Traditional (`zh_hant`)

| Property       | Value                                |
|----------------|--------------------------------------|
| Language       | Chinese (Traditional) (`zh_hant`)    |
| Total Samples  | 2,214,304                            |
| Train          | 1,772,280 samples (77 files)         |
| Test           | 221,867 samples (77 files)           |
| Validation     | 220,157 samples (77 files)           |

Total Data Storage: 5.45TB

## Reference(s):
The synthetic data generation pipeline is based on SynthDoG from the Donut project, with substantial modifications to support additional languages, custom rendering effects, structured bounding-box annotations (word/line/paragraph levels with reading-order graphs), and HDF5 output.
## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse.   
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns here.  

## Acknowledgements

The synthetic data generation pipeline is based on [SynthDoG](https://github.com/clovaai/donut/tree/master/synthdog) from the Donut project, with substantial modifications to support additional languages, custom rendering effects, structured bounding-box annotations (word/line/paragraph levels with reading-order graphs), and HDF5 output.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{chesler2026ocr_synthetic_multilingual,
  title   = {{OCR-Synthetic-Multilingual-v1}},
  author  = {Chesler, Ryan},
  year    = {2026},
  publisher = {NVIDIA},
  url     = {https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1},
  note    = {Synthetically generated multilingual OCR dataset built on a heavily modified SynthDoG pipeline}
}
```
