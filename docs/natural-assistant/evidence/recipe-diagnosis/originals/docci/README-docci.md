---
annotations_creators:
- expert-generated
- crowdsourced
language:
- en
language_creators:
- other
license:
- cc-by-4.0
multilinguality:
- monolingual
pretty_name: DOCCI
size_categories:
- 10K<n<100K
source_datasets:
- original
tags: []
task_categories:
- text-to-image
- image-to-text
task_ids:
- image-captioning
---

# Dataset Card for DOCCI

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks](#supported-tasks)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://google.github.io/docci
- **Paper:** [arXiv](https://arxiv.org/pdf/2404.19753)
- **Data Explorer:** [Check images and descriptions](https://google.github.io/docci/viz.html?c=&p=1)
- **Point of Contact:** docci-dataset@google.com
- **Report an Error:** [Google Forms](https://forms.gle/v8sUoXWHvuqrWyfe9)

### Dataset Summary

DOCCI (Descriptions of Connected and Contrasting Images) is a collection of images paired with detailed descriptions. The descriptions explain the key elements of the images, as well as secondary information such as background, lighting, and settings. The images are specifically taken to help assess the precise visual properties of images. DOCCI also includes many related images that vary in having key differences from the others. All descriptions are manually annotated to ensure they adequately distinguish each image from its counterparts.

### Supported Tasks

Text-to-Image and Image-to-Text generation

### Languages

English

## Dataset Structure

### Data Instances

```
{
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1536x2048>,
    'example_id': 'qual_dev_00000',
    'description': 'An indoor angled down medium close-up front view of a real sized stuffed dog with white and black colored fur wearing a blue hard hat with a light on it. A couple inches to the right of the dog is a real sized black and white penguin that is also wearing a blue hard hat with a light on it. The dog is sitting, and is facing slightly towards the right while looking to its right with its mouth slightly open, showing its pink tongue. The dog and penguin are placed on a gray and white carpet, and placed against a white drawer that has a large gray cushion on top of it. Behind the gray cushion is a transparent window showing green trees on the outside.'
}
```

### Data Fields

Name | Explanation
--- | ---
`image`       | PIL.JpegImagePlugin.JpegImageFile               
`example_id`  | The unique ID of an example follows this format: `<SPLIT_NAME>_<EXAMPLE_NUMBER>`.
`description` | Text description of the associated image.

### Data Splits

Dataset | Train | Test | Qual Dev | Qual Test
---| ---: | ---: | ---: | ---: 
DOCCI     | 9,647 | 5,000 | 100 | 100 
DOCCI-AAR | 4,932 | 5,000 | --  | --


## Dataset Creation

### Curation Rationale

DOCCI is designed as an evaluation dataset for both text-to-image (T2I) and image-to-text (I2T) generation. Please see our paper for more details.

### Source Data

#### Initial Data Collection

All images were taken by one of the authors and their family.

### Annotations

#### Annotation process

All text descriptions were written by human annotators.
We do not rely on any automated process in our data annotation pipeline.
Please see Appendix A of [our paper](https://arxiv.org/pdf/2404.19753) for details about image curation.

### Personal and Sensitive Information

We manually reviewed all images for personally identifiable information (PII), removing some images and blurring detected faces, phone numbers, and URLs to protect privacy.
For text descriptions, we instructed annotators to exclude any PII, such as people's names, phone numbers, and URLs.
After the annotation phase, we employed automatic tools to scan for PII, ensuring the descriptions remained free of such information.

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

### Licensing Information

CC BY 4.0

### Citation Information

```
@inproceedings{OnoeDocci2024,
  author        = {Yasumasa Onoe and Sunayana Rane and Zachary Berger and Yonatan Bitton and Jaemin Cho and Roopal Garg and
    Alexander Ku and Zarana Parekh and Jordi Pont-Tuset and Garrett Tanzer and Su Wang and Jason Baldridge},
  title         = {{DOCCI: Descriptions of Connected and Contrasting Images}},
  booktitle     = {ECCV},
  year          = {2024}
}
```