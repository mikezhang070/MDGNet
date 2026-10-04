# MDGNet

**MDGNet: Multi-Domain Graph-Spectral Network with Variable-Level Routing for Long-Term Time Series Forecasting**

MDGNet is a Transformer-based architecture for **long-term multivariate time series forecasting**. Built on the inverted (variate-as-token) encoding paradigm, it combines **graph-domain** and **spectral-domain** modeling and routes individual variables to the branch best suited to them, yielding a flexible and expressive model that handles datasets with hundreds of variates.

<p align="center">
<em>Graph (inter-variate structure) + Spectral (frequency structure) + Variable-Level Routing → accurate long-horizon forecasts.</em>
</p>

---

## Overview

Conventional Transformers treat each *time step* as a token and rely on attention to discover cross-time dependencies. Following the inverted paradigm, MDGNet instead treats each *variate* as a token, so attention operates across variables while a feed-forward network learns the series representation of each variate.

MDGNet goes further with **multi-domain modeling**:

- **Graph domain** — a graph-convolutional branch explicitly models inter-variate dependencies through a learned adjacency matrix and graph diffusion.
- **Spectral domain** — a frequency-processing branch decomposes the series into trend and seasonal components and encodes them with complex-valued operations in the Fourier (STFT) domain.
- **Recurrent refinement** — a multi-domain gated recurrent block (MGRBlock) captures joint variable/feature interactions.

A **variable-level routing** mechanism (`--num_gcn_channels`) splits the set of variates: a subset is processed by the graph branch while the rest flow through the spectral branch, and the two streams are fused by concatenation. This lets each variate be modeled by the domain most appropriate for it.

## Quick Start

1. Install PyTorch and the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download the datasets (see [Datasets](#datasets)) and place them under `./dataset/` following the paths used by the scripts (e.g. `./dataset/ETT-small/ETTh1.csv`, `./dataset/Flight/Flight.csv`).

3. Train and evaluate. For example, multivariate forecasting on the Flight dataset:

   ```bash
   bash scripts/multivariate_forecasting/Weather/MDGNet.sh
   ```


## Datasets

| Dataset | Description | Download |
|---------|-------------|----------|
| **ETT** (`ETTh1/2`, `ETTm1/2`) | Electricity Transformer Temperature, hourly / 15-min. | [ETDataset (GitHub)](https://github.com/zhouhaoyi/ETDataset) |
| **ECL** (Electricity) | Hourly electricity consumption of 321 clients (`electricity.csv`). | [Time-Series-Library datasets (Google Drive)](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) · [Baidu Pan (pwd `i9iy`)](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy) |
| **Exchange** | Daily exchange rates of 8 countries, 1990–2010 (`exchange_rate.csv`). | Same as above |
| **Weather** | 21 meteorological indicators recorded every 10 min (`weather.csv`). | Same as above |
| **Traffic** | Hourly road occupancy rates of 862 sensors (`traffic.csv`). | Same as above |
| **Solar** | Solar power of 137 PV plants (`solar_AL.txt`). | Same as above |
| **Flight** | Hourly flight dataset (7 channels, target `UUEE`), described in the PatchWaveNet paper — Z. Huang, F. Zhang, Y. Liu, *EAAI* 154 (2025). | [Paper (DOI)](https://doi.org/10.1016/j.engappai.2025.110964) |

The standard benchmark datasets (ECL, Exchange, Weather, Traffic, Solar) are also available from the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) repository, which hosts the well pre-processed versions.

## License

This project is released under the [MIT License](./LICENSE).

## Contact

For questions or collaboration, please open an issue in this repository or contact the maintainer.
