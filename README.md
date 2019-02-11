# Unsupervised Tweet Clustering with DEC
The aim of this project is to clustering the unlabelled tweets with dec and I inspired by: "Unsupervised Deep Embedding for Clustering Analysis" [(Xie et al, ICML 2016)](https://arxiv.org/pdf/1511.06335.pdf).

Before the dec I applied natural language processing techniques to preprocess to json type data. Then I use dec and I use the dec because that has more accuracy on clustering.
## File Structure
```
.
+-- tweets.json	
+-- hey_thats_your.py
+-- keras_dec.py
+-- requirements.txt
+-- README.md
+-- .gitignore

```

## Prerequisites
If you want a virtual enviroment, firstly create it:

`python -m venv virenv`

than activate:

`source virenv/bin/activate`

finally install packages:

`pip install -r requirements.txt`

## Usage
Run:

`python hey_thats_your.py`

If we want to run on CPU we set the local variable `set CUDA_VISIBLE_DEVICES=-1` then run it otherwise run on GPU.
