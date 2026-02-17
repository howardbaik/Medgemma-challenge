# EchoNet-MIMIC VQA Dataset

## Purpose

The EchoNet-MIMIC Visual Question Answering (VQA) dataset is designed to evaluate AI models' ability to understand and interpret echocardiography videos through natural language questions and answers. This dataset combines echocardiogram video data from the MIMIC-IV-Echo database with expert-curated and edited questions covering various aspects of cardiac imaging interpretation (i.e., disease diagnosis, anatomical view recognition, measurement grading, and descriptive analysis).

## Dataset Statistics

The dataset contains **258 questions** (Single Visual QA) across multiple task categories:

### Content Types
- **Diagnosis**: 226 questions
- **View**: 32 questions

### Question Types
- **Binary**: 113 questions - Yes/No questions requiring binary responses
- **Diagnosis**: 53 questions - Specific diagnostic reasoning questions
- **Descriptive**: 46 questions - Open-ended descriptive questions about cardiac features
- **View**: 32 questions - Echocardiographic view identification
- **Grading**: 14 questions - Severity grading questions (like mild/moderate/severe)

## Task Types

The dataset evaluates the following clinical capabilities:

1. **Disease Detection and Diagnosis** - Identifying pathological conditions from echocardiogram videos
2. **View Recognition** - Classifying standard echocardiographic views (e.g., parasternal long axis, apical four-chamber)
3. **Severity Grading** - Assessing the severity of cardiac abnormalities
4. **Descriptive Analysis** - Providing detailed descriptions of cardiac structures and functions

## Folder Structure

### Data File Structure

The main CSV file (`MIMIC_Echo_1qa_SDE_vFINAL_share_d20260210_111554.csv`) contains the following columns:

- `question_id`: Unique identifier for each question
- `study_id`: MIMIC study identifier
- `target_id`: Target concept/disease identifier
- `target_name`: Name of the target concept
- `content_type`: Type of content (Diagnosis/View)
- `question_type`: Format of question (Binary/Diagnosis/Descriptive/View/Grading)
- `explanation`: Rationale from the clinical report
- `dicom_path`: Path to source DICOM file (Please download and save your workspace)
- `mp4_path`: Path to converted MP4 video (you can make mp4 video with `Medgemma-challenge/EchoNet-MIMIC_VQA/0_convert_Dicom_to_AVI_save.py`)
- `disease_label`: Disease category label
- `report`: Full clinical echocardiogram report (Raw report from MIMIC)
- `Final_Question`: The question text
- `Final_option_A/B/C/D`: Multiple choice options
- `Final_correct_option`: Correct answer
- `review_timestamp`: Timestamp of quality review

## Sample

A sample question-answer pair with its corresponding echocardiogram video frame is shown in [assets/Demo_1.png](assets/Demo_1.png).

## Source Data

The echocardiogram videos are sourced from the MIMIC-IV-Echo database. To download the source DICOM files:

**MIMIC-IV-Echo Homepage**: https://physionet.org/content/mimic-iv-echo/0.1/

### MIMIC-IV-Echo File Structure
```
/YOUR workspace/.../DICOM/MIMIC
├── p10/
│   └── p10690270/
│       ├── s95240362/
│       │   ├── 95240362_0004.dcm
│       │   └── ...
│       └── s90045402/
│           ├── 90045402_0001.dcm
│           └── ...
└── p19/
    └── p19425623/
        └── s90267113/
            ├── 90267113_0001.dcm
            └── ...
```

**Note**: Access to MIMIC-IV-Echo requires credentialed access through PhysioNet.