# A novel Deep Learning approach for one-step Conformal Prediction approximation
This repository is in support of our paper "A Confident Deep Learning loss function for one-step Conformal Prediction approximation". We show both the code used for results generation, and the specific results presented in the paper. For method descriptions and results interpretation, please see our paper here: https://link.springer.com/article/10.1007/s10472-023-09849-y

Deep Learning predictions with measurable confidence are increasingly desirable for real-world problems, especially in high-risk settings. The Conformal Prediction (CP) framework is a versatile solution that guarantees a maximum error rate given minimal constraints. In this paper, we propose a novel conformal loss function that approximates the traditionally two-step CP approach in a single step. By evaluating and penalising deviations from the stringent expected CP output distribution, a Deep Learning model may learn the direct relationship between the input data and the conformal p-values. We carry out a comprehensive empirical evaluation to show our novel loss function’s competitiveness for seven binary and multi-class prediction tasks on five benchmark datasets. On the same datasets, our approach achieves significant training time reductions up to 86% compared to Aggregated Conformal Prediction (ACP), while maintaining comparable approximate validity and predictive efficiency.


## Environment setup
The code is written in Python 3.9. To automatically set up a Conda environment with all needed packages, run ```conda install --file conda_requirements_py39.txt``` in the root directory


## Citation
Meister, J. A., Nguyen, K. A., Kapetanakis, S., & Luo, Z. (2023). A novel deep learning approach for one-step conformal prediction approximation. *Annals of Mathematics and Artificial Intelligence*, 1-28.


@article{meister2023novel,
  title={A novel deep learning approach for one-step conformal prediction approximation},
  author={Meister, Julia A and Nguyen, Khuong An and Kapetanakis, Stelios and Luo, Zhiyuan},
  journal={Annals of Mathematics and Artificial Intelligence},
  pages={1--28},
  year={2023},
  publisher={Springer}
}
