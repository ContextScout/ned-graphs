Code and Dataset for Named Entity Disambiguation using Deep Learning on Graphs
==============================================================================

This repository contains the code and dataset for the paper "Named Entity Disambiguation using Deep Learning on Graphs". The full paper can be found [here](https://arxiv.org/pdf/1810.09164.pdf).


Installation
------------
The main requirements are installed with:

```bash
virtualenv --python=/usr/bin/python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

Download the glove vector files

```bash
cd data
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
echo "2196017 300" | cat - glove.840B.300d.txt > glove_2.2M.txt
cd ..
```

One must also unzip the mapping of the wikidata nodes

```bash
cd data
bunzip2 x*
cat x* > wikidata_items.csv
cd ..
```


Running the models
------------------

In order to train the system

```bash
cd <MODEL_NAME>
python -m wikidata_query.train
```

Similarly for testing
```bash
cd <MODEL_NAME>
python -m wikidata_query.test
```


Citing the paper
----------------
```code

@ARTICLE{2018arXiv181009164C,
   author = {{Cetoli}, A. and {Akbari}, M. and {Bragaglia}, S. and {O'Harney}, A.~D. and 
	{Sloan}, M.},
    title = "{Named Entity Disambiguation using Deep Learning on Graphs}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1810.09164},
 primaryClass = "cs.CL",
 keywords = {Computer Science - Computation and Language},
     year = 2018,
    month = oct,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv181009164C},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```