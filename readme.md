# 🧠 Differential GABA Dynamics Across Brain Functional Networks in Autism

This repository contains the **data and analysis code** related to the study entitled:

> **"Differential GABA dynamics across brain functional networks in autism."**

The work investigates alterations in **GABAergic dynamics** across large-scale functional brain networks in individuals with **autism spectrum disorder (ASD)**.

Both the dataset and code are provided for **research transparency and reproducibility**.  
The contents may be updated as the manuscript undergoes further revision and peer review.

---

## 📂 Repository Structure

```text
code/
├── source_localization/        # EEG source modeling and network-level extraction
└── PAC_calculate/              # PAC computation, statistical analysis, and visualization

data/                           # PAC values and statistical results (stored directly here)

info/
└── all_paticipant_info.xlsx    # Participant demographic and assessment information

```
---

## 📊 Data

All processed data are stored in the `data/` directory and include:

- **PAC values**  
  Network-level PAC measures computed within and between large-scale functional networks.

- **Statistical results**  
  Corresponding **t-values** and **p-values** from group- and condition-level comparisons.

These files provide the quantitative basis for the main results reported in the study.

---

## 💻 Code

All analysis scripts are stored in the `code/` directory and are organized as follows:

- `code/source_localization/`  
  - EEG preprocessing and source reconstruction  
  - Mapping sensor-level EEG data to cortical functional networks  
  - Extracting time series for subsequent PAC analysis  

- `code/PAC_calculate/`  
  - Computing PAC across relevant frequency bands  
  - Deriving within-network and between-network PAC connectivity  
  - Performing statistical analyses and generating visualizations of PAC alterations  


---

## 📩 Contact
Encrypted files can be decrypted using GPG with a password provided upon request.
For questions, comments, or collaboration inquiries, please contact:

**Qiyun Huang**  
School of Future Technology, South China University of Technology, China 
📧 [huangqiyun@pazhoulab.cn](mailto:huangqiyun@pazhoulab.cn)

**Di Chen**  
Research Center for Brain-Computer Interface, Pazhou Lab, Guangzhou, China
📧 [dididichen@outlook.com](mailto:dididichen@outlook.com)
