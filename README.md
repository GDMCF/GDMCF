## GDMCF is a Graph-based Diffusion Model for Collaborative Filtering


## Environment

- Anaconda 3
- Python 3.8.10
- PyTorch 1.12.0
- NumPy 1.22.3

## Data

The experimental data are in the `./datasets` folder, including Amazon-Book and Yelp. Note that the results on ML-1M differ from those reported in CODIGEM, owing to different data processing procedures. CODIGEM did not sort and split the training/testing sets according to timestamps; however, temporal splitting aligns better with real-world testing.

## Usage

### Training

```bash
cd project-name

