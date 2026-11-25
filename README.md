# \# VUS-Guard

# \### Uncertainty-Aware Prediction of Drug Resistance-Associated Mutations in \*Mycobacterium tuberculosis\*

# 

# <p align="center">

# &#x20; <strong>Developed by Ishwari Bhawalkar</strong><br>

# &#x20; Biotechnology | Machine Learning | Bioinformatics

# </p>

# 

# <p align="center">

# &#x20; <a href="https://ishwari-bhawalkar-vusguard.streamlit.app">Live Application</a> |

# &#x20; <a href="https://github.com/IshwariBhawalkar/vus-guard">GitHub Repository</a>

# </p>

# 

# \---

# 

# \## Overview

# 

# \*\*VUS-Guard\*\* is an uncertainty-aware machine learning research prototype designed to analyse genetic mutations associated with drug resistance in \*Mycobacterium tuberculosis\*.

# 

# Drug-resistant tuberculosis remains a major global health challenge. Identifying mutations associated with resistance can support the interpretation of genomic data and contribute to research in precision medicine.

# 

# VUS-Guard uses machine learning to analyse mutation-related information and classify variants into three possible outcomes:

# 

# \- \*\*Resistant:\*\* The model predicts an association with drug resistance.

# \- \*\*Susceptible:\*\* The model predicts an association with drug susceptibility.

# \- \*\*Abstain:\*\* The model does not make a confident prediction.

# 

# Unlike a conventional binary classifier, VUS-Guard incorporates an abstention mechanism to avoid forcing predictions when the available evidence is insufficient.

# 

# \## Key Features

# 

# \- Machine learning-based mutation classification.

# \- Drug-specific and gene-specific variant analysis.

# \- Uncertainty-aware prediction with an abstention mechanism.

# \- Model validation using held-out data.

# \- Sanity checks to evaluate model behaviour.

# \- Interactive Streamlit application.

# \- Visualisation of model performance and prediction results.

# 

# \## Technology Stack

# 

# | Category | Technologies |

# |---|---|

# | Programming Language | Python |

# | Data Processing | Pandas, NumPy |

# | Machine Learning | Scikit-learn |

# | Bioinformatics | Biopython |

# | Model Persistence | Joblib |

# | Visualisation | Matplotlib |

# | Web Application | Streamlit |

# | Data Format | CSV, Excel |

# 

# \## Model Performance and Validation

# 

# The model was evaluated using multiple validation approaches to examine its predictive performance and generalisation behaviour.

# 

# | Evaluation Metric | Observed Result |

# |---|---:|

# | Missense ROC-AUC | 0.962 |

# | Position-held-out mean AUC | 0.965 |

# | Label permutation mean AUC | 0.477 |

# | Leave-gene-out mean AUC | 0.545 |

# 

# \### Understanding the Results

# 

# \- \*\*Missense ROC-AUC:\*\* Indicates strong discrimination on the evaluated missense mutation data.

# \- \*\*Position-held-out validation:\*\* Tests performance on mutation positions excluded during training.

# \- \*\*Label permutation:\*\* Checks whether the model performs better than expected when labels are shuffled.

# \- \*\*Leave-gene-out validation:\*\* Evaluates how well the model generalises to genes not represented in training.

# 

# The lower leave-gene-out performance highlights an important limitation: predictions for previously unseen genes require additional evidence and careful interpretation.

# 

# \## Uncertainty-Aware Prediction

# 

# A central feature of VUS-Guard is its ability to abstain from making a prediction.

# 

# When the model encounters insufficient supporting evidence or an uncertain prediction, it can return an \*\*ABSTAIN\*\* result instead of automatically assigning a resistant or susceptible label.

# 

# This is particularly important in genomic interpretation, where an incorrect prediction may lead to misleading conclusions.

# 

# \*\*An abstention is not a diagnosis and does not indicate that a mutation is resistant or susceptible.\*\*

# 

# \## Data Source

# 

# The project uses the WHO 2023 catalogue of mutations associated with drug resistance in \*Mycobacterium tuberculosis\* as its reference data source.

# 

# The catalogue provides mutation-level information used in preparing the processed dataset for model development and evaluation.

# 

# \## Application

# 

# The VUS-Guard application is built using Streamlit and provides an interactive interface for exploring mutation predictions.

# 

# \*\*Try the application:\*\*

# 

# \[Launch VUS-Guard](https://ishwari-bhawalkar-vusguard.streamlit.app)

# 

# The application allows users to:

# \- Select a drug.

# \- Enter a gene and mutation.

# \- View the model's prediction.

# \- Examine prediction confidence and abstention behaviour.

# \- Review model reliability information.

# 

# \## Project Structure

# 

# ```text

# vus-guard/

# │

# ├── app/

# │   └── app.py

# │

# ├── data/

# │   └── processed/

# │

# ├── models/

# │   └── vus\_guard.joblib

# │

# ├── reports/

# │   ├── fig1\_validation.png

# │   ├── fig2\_abstention.png

# │   ├── fig3\_vus\_predictions.png

# │   ├── metrics.json

# │   ├── o

