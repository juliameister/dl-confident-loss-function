# A novel Deep Learning approach for one-step Conformal Prediction approximation
This repository is in support of our paper "A Confident Deep Learning loss function for one-step Conformal Prediction approximation". We show both the code used for results generation, and the specific results presented in the paper. For method descriptions and results interpretation, please see our paper here: https://arxiv.org/abs/2207.12377

Deep Learning predictions with measurable confidence are increasingly desirable for real-world problems, especially in high-risk settings. The Conformal Prediction (CP) framework is a versatile solution that automatically guarantees a maximum error rate. However, CP suffers from computational inefficiencies that limit its application to large-scale datasets. We propose a novel conformal loss function that approximates the traditionally two-step CP approach in a single step. By evaluating and penalising deviations from the stringent expected CP output distribution, a Deep Learning model may learn the direct relationship between input data and conformal p-values. Our approach achieves significant training time reductions up to 86% compared to Aggregated Conformal Prediction (ACP), an accepted CP approximation variant. In terms of approximate validity and predictive efficiency, we carry out a comprehensive empirical evaluation to show our novel loss function's competitiveness with ACP on the well-established MNIST dataset. 


## Environment setup
The code is written in Python 3.9. To automatically set up a Conda environment with all needed packages, run ```conda install --file conda_requirements_py39.txt``` in the root directory


## Citation
Julia A. Meister, Khuong An Nguyen, Stelios Kapetanakis, and Zhiyuan Luo. "A Confident Deep Learning loss function for one-step Conformal Prediction approximation." (2022). https://arxiv.org/abs/2207.12377


@misc{meister2022confident,
    title={A Confident Deep Learning loss function for one-step Conformal Prediction approximation},
    author={Julia A. Meister and Khuong An Nguyen and Stelios Kapetanakis and Zhiyuan Luo},
    year={2022},
    eprint={2207.12377},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
