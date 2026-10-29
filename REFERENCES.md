# References & Methodological Grounding

This project's data preprocessing, cycle extraction, disaggregation model design, evaluation protocols, and practical recommendations are grounded in peer-reviewed academic literature in non-intrusive load monitoring (NILM) and energy disaggregation.

---

### Core Methodological References

1. **Murray, D., Stankovic, L., & Stankovic, V. (2017).**  
   *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study.*  
   *Scientific Data*, 4, 160122.  
   DOI: [10.1038/sdata.2016.122](https://doi.org/10.1038/sdata.2016.122) · PMC: [PMC5315495](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5315495/)  
   *Role in this project:* Primary documentation for the REFIT dataset structure, physical metering setup (CurrentCost 433 MHz RF clamps and IAM plug meters rated up to 4 kW), 8-second nominal sampling, household metadata, missing data characteristics, and official appliance channel allocations (Table 3).

2. **Batra, N., Kelly, J., Parson, O., Dutta, H., Knottenbelt, W., Rogers, A., Singh, A., & Srivastava, M. (2014).**  
   *NILMTK: An Open Source Toolkit for Non-intrusive Load Monitoring.*  
   *Proceedings of the 5th International Conference on Future Energy Systems (ACM e-Energy)*, 265–276.  
   arXiv: [1404.3878](https://arxiv.org/abs/1404.3878)  
   *Role in this project:* Motivates standardized preprocessing workflows, resampling to consistent 1-minute time bins, handling of appliance state boundaries, and evaluation metrics (MAE, Normalized Disaggregation Error [NDE], and classification metrics).

3. **Zhang, C., Zhong, M., Wang, Z., Goddard, N., & Sutton, C. (2018).**  
   *Sequence-to-Point Learning with Neural Networks for Non-Intrusive Load Monitoring.*  
   *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1), 2604–2611.  
   URL: [AAAI Publication](https://ojs.aaai.org/index.php/AAAI/article/view/11873)  
   *Role in this project:* Motivates our window-to-point disaggregation formulation: using a context window of aggregate power centred on the target minute ($\pm 30$ min) to infer the middle point's appliance power. We adapt this principle to gradient boosted decision trees (`HistGradientBoostingRegressor`) with explicit windowed summary statistics.

4. **Kelly, J., & Knottenbelt, W. (2015).**  
   *Neural NILM: Deep Neural Networks Applied to Energy Disaggregation.*  
   *Proceedings of the 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments (BuildSys)*, 55–64.  
   arXiv: [1507.06594](https://arxiv.org/abs/1507.06594)  
   *Role in this project:* Grounding for leave-one-household-out (unseen home) evaluation, thresholding continuous predictions into active appliance states, and analyzing appliance activation signatures against baseline whole-house loads.

5. **Murray, D., Stankovic, L., Stankovic, V., & Labeodan, T. (2019).**  
   *Transferability of neural network approaches for low-rate energy disaggregation.*  
   *Building and Environment*, 160, 106203.  
   URL: [Strathprints 66112](https://strathprints.strath.ac.uk/66112)  
   *Role in this project:* Justifies our Leave-One-Household-Out (LOHO) cross-validation protocol, highlighting the substantial drop in generalization when evaluating models on unseen households due to machine make/model diversity and varying wash habits.

6. **Klemenjak, C., Makonin, S., & Elmenreich, W. (2020).**  
   *Towards Comparability in Non-Intrusive Load Monitoring: On Data and Performance Evaluation.*  
   *IEEE Power & Energy Society Innovative Smart Grid Technologies Conference (ISGT)*.  
   arXiv: [2001.07708](https://arxiv.org/abs/2001.07708)  
   *Role in this project:* Guides transparent documentation of data cleaning, explicit reporting of test-set sizes, gap handling, and metric definitions to prevent optimistic or incomparable benchmark claims.

7. **Klemenjak, C., Reinhardt, A., Pereira, L., Makonin, S., Dahlbom, M., & Elmenreich, W. (2020).**  
   *On Metrics to Assess the Transferability of Machine Learning Models in Non-Intrusive Load Monitoring.*  
   arXiv: [1912.06200](https://arxiv.org/abs/1912.06200)  
   *Role in this project:* Motivates multi-metric assessment across unseen houses (MAE, RMSE, SAE, NDE, F1, cycle recall) rather than relying on a single aggregate metric that can flatter majority-off predictions.

8. **D’Incecco, M., Squartini, S., & Zhong, M. (2020).**  
   *Transfer Learning for Non-Intrusive Load Monitoring.*  
   *IEEE Transactions on Smart Grid*, 11(2), 1419–1429.  
   arXiv: [1902.08835](https://arxiv.org/abs/1902.08835)  
   *Role in this project:* Background on domain shifts between different building stocks, informing our discussion on the obstacles of transferring UK-trained models to Indian domestic households.

9. **Huber, P., Calatroni, A., Rumsch, A., & Paice, A. (2022).**  
   *Towards trustworthy Energy Disaggregation: A review of challenges, methods and perspectives for Non-Intrusive Load Monitoring.*  
   *Energy and Buildings*, 271, 112286.  
   arXiv: [2207.02009](https://arxiv.org/abs/2207.02009)  
   *Role in this project:* Methodological grounding for our failure-case analysis, identifying signature overlap with concurrent heavy appliances (e.g. dishwashers, kettles) and sensor dropouts as primary error drivers.

10. **Yue, Z., Witzig, C. R., Jorde, D., & Jacobsen, H. A. (2020).**  
    *BERT4NILM: A Bidirectional Transformer Model for Non-Intrusive Load Monitoring.*  
    *ACM BuildSys*, 89–98.  
    URL: [TUM Portal](https://portal.fis.tum.de/en/publications/bert4nilm-a-bidirectional-transformer-model-for-non-intrusive-load-monitoring/)  
    *Role in this project:* Documented as a potential future architecture for capturing long-range bidirectional dependencies, while acknowledging that lightweight gradient boosting remains preferable for fast, explainable CPU-based utility pilots.
