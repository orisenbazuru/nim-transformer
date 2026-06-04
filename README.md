# Nim Transformer Game

This repository contains an educational Nim game and two model-training notebooks used to teach machine learning ideas through a small, visual game.

The core teaching arc is:

```text
Nim board -> bit representation -> balance / Nim-sum prediction -> attention-based move model -> browser game
```

The browser app lets a human play Nim against a small transformer-style model. The learning materials explain how the model first learns the board balance, then uses attention to choose which pile to change and how many stones should remain.

## App Website

GitHub Pages app:

```text
https://orisenbazuru.github.io/nim-transformer/
```


## Repository Structure

- [`docs/index.html`](docs/index.html): static browser app for GitHub Pages.
- [`docs/model_weights.json`](docs/model_weights.json): exported model weights used by the browser app.
- [`docs/.nojekyll`](docs/.nojekyll): tells GitHub Pages to serve the static files directly.
- [`01_train_xor_sum_predictor_clean.ipynb`](01_train_xor_sum_predictor_clean.ipynb): trains the balance / Nim-sum predictor.
- [`02_train_nim_transformer_pipeline_clean.ipynb`](02_train_nim_transformer_pipeline_clean.ipynb): trains and evaluates the attention-based Nim move model.
- [`prepare_github_pages.py`](prepare_github_pages.py): exports trained PyTorch weights into the browser-readable JSON format.

## Notebook 1: Balance / Nim-Sum Predictor

Notebook:

[`01_train_xor_sum_predictor_clean.ipynb`](01_train_xor_sum_predictor_clean.ipynb)

Purpose:

- Generate Nim board examples.
- Convert pile sizes into bit vectors.
- Train a two-layer neural network with ReLU to predict the Nim-sum / balance bits.
- Save the trained checkpoint for downstream use.

Default model:

```text
board bits -> Linear -> ReLU -> Linear -> Sigmoid -> balance bits
```

The notebook is configurable for:

- number of piles
- maximum stones per pile
- hidden layer size
- training epochs

## Notebook 2: Nim Transformer Move Model

Notebook:

[`02_train_nim_transformer_pipeline_clean.ipynb`](02_train_nim_transformer_pipeline_clean.ipynb)

Purpose:

- Load the balance predictor from notebook 1.
- Generate training examples for optimal Nim moves.
- Train a small transformer-style move model.
- Evaluate the full pipeline using predicted balance bits.
- Visualize the model's inner decision process.
- Run interactive notebook gameplay.

Pipeline:

```text
board bits
  -> predicted balance bits
  -> balance query + pile keys / values
  -> attention over piles
  -> selected pile
  -> target-size prediction
  -> updated board
```

The notebook includes helper plots for:

- pile stones as circle visuals
- bit grids and balance rows
- model attention over piles
- predicted target-size bits
- before/after board updates


## Environment Setup

The notebooks use Python, PyTorch, NumPy, Matplotlib, Jupyter, and ipywidgets.

### Option A: Conda

Create and activate an environment:

```bash
conda create -n nim-transformer python=3.10 -y
conda activate nim-transformer
```

Install the main packages:

```bash
conda install -c conda-forge numpy matplotlib jupyterlab ipywidgets nbformat -y
conda install pytorch torchvision torchaudio -c pytorch -y
```

Start Jupyter:

```bash
jupyter lab
```

Then open:

```text
01_train_xor_sum_predictor_clean.ipynb
02_train_nim_transformer_pipeline_clean.ipynb
```

### Option B: environment.yml

If you use the included environment file:

```bash
conda env create -f environment.yml
conda activate nim-transformer
jupyter lab
```

## Training and Export Flow

Run the notebooks in this order:

1. `01_train_xor_sum_predictor_clean.ipynb`
2. `02_train_nim_transformer_pipeline_clean.ipynb`

After training, export the move model for the static app:

```bash
python3 prepare_github_pages.py
```

This updates:

```text
docs/model_weights.json
```

Commit that JSON file along with the app.

## GitHub Pages Deployment

In GitHub:

1. Open the repository.
2. Go to **Settings**.
3. Go to **Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Set:
   - Branch: `main`
   - Folder: `/docs`
6. Save.

GitHub will publish the app at:

```text
https://orisenbazuru.github.io/nim-transformer/
```

## Notes

- The deployed app is fully static.
- GitHub Pages does not run Python, Flask, or PyTorch.
- The browser app runs the trained model forward pass directly in JavaScript using `docs/model_weights.json`.
- Retraining happens locally in the notebooks; publishing only needs the exported JSON weights.
