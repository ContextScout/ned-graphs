Code and Dataset for Named Entity Disambiguation using Deep Learning on Graphs
==============================================================================


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
echo "2196017 300" | cat _ glove.840B.300d > glove_2.2M.txt
cd ..
```

One must also unzip the mapping of the wikidata nodes

```bash
cd data
bunzip2 x*
cat x* > wikidata_items.csv
cd ..
```