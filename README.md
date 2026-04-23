# BIG-DATA-FINAL-PROYECT-2025-01
-3-
Healthcare Data Intelligence 
Overview

This project focuses on building a data intelligence pipeline for the healthcare domain. The objective is to transform a raw dataset into a structured and normalized format that supports clustering, recommendation, and graph-based analysis.

The system includes data ingestion, preprocessing, schema design, and the creation of relational tables to enable advanced analytical tasks.

Problem Statement

Healthcare datasets are often unstructured and difficult to analyze. This project addresses the need for a clean and normalized data model that enables pattern discovery across patient profiles, medical conditions, treatments, and hospital outcomes.

Product Question

Is it possible to cluster patient profiles and treatment pathways to identify patterns between medical conditions, medication types, and hospital admission outcomes?
#### Data Dictionary Draft

| Column Name        | Description                                       | Data Type   |
|:-------------------|:--------------------------------------------------|:------------|
| `Name`             | Patient's name                                    | string      |
| `Age`              | Age of the patient                                | integer     |
| `Gender`           | Gender of the patient (Male, Female)              | categorical |
| `Blood Type`       | Blood type of the patient                         | categorical |
| `Medical Condition`| Diagnosed medical condition                       | categorical |
| `Date of Admission`| Date of patient admission                         | datetime    |
| `Doctor`           | Attending doctor's name                           | string      |
| `Hospital`         | Hospital name                                     | string      |
| `Insurance Provider`| Insurance company                                 | categorical |
| `Billing Amount`   | Total billing amount                              | float       |
| `Room Number`      | Room assigned to the patient                      | integer     |
| `Admission Type`   | Type of admission (Emergency, Elective, Urgent)   | categorical |
| `Discharge Date`   | Date of patient discharge                         | datetime    |
| `Medication`       | Prescribed medication                             | categorical |
| `Test Results`     | Results of medical tests                          | categorical |
