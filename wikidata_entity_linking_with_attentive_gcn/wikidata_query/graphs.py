import requests

from wikidata_query.wikidata_items import wikidata_items

query_nn_back = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?item0 ?rel ?item1 WHERE {
  ?item0 rdfs:label "%s"@en .
  ?item1 ?rel ?item0 .
  FILTER regex (str(?item1), '^((?!statement).)*$') .
  FILTER regex (str(?item1), '^((?!https).)*$') .
} LIMIT 1500
'''

query_nn2_back = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?item0 ?rel ?item1 WHERE {
  ?item rdfs:label "%s"@en .
  ?item0 ?r ?item .
  ?item1 ?rel ?item0 .
  FILTER regex (str(?item0), '^((?!statement).)*$') .
  FILTER regex (str(?item0), '^((?!https).)*$') .
  FILTER regex (str(?item1), '^((?!statement).)*$') .
  FILTER regex (str(?item1), '^((?!https).)*$') .
} limit 1000
'''

query_nn3_back = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?item0 ?rel ?item1 WHERE {
  ?item rdfs:label "%s"@en .
  ?item_nn ?r ?item .
  ?item0 ?r2 ?item_nn .
  ?item1 ?rel ?item0 .
} limit 1000
'''

query_nn_forw = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?item0 ?rel ?item1  WHERE {
  ?item0 rdfs:label "%s"@en .
  ?item0 ?rel ?item1 .
  FILTER regex (str(?item1), '^((?!statement).)*$') .
  FILTER regex (str(?item1), '^((?!https).)*$') .
} limit 5000
'''

query_nn2_forw = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?item0 ?rel ?item1 WHERE {
  ?item rdfs:label "%s"@en .
  ?item ?r ?item0 .
  ?item0 ?rel ?item1 .
  FILTER regex (str(?item0), '(statement)') .  # two hops only for in-line statements
  FILTER regex (str(?item1), '^((?!statement).)*$') .
  FILTER regex (str(?item1), '^((?!https).)*$') .
} limit 1000
'''

query_nn3_forw = '''
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?item0 ?rel ?item1 WHERE {
  ?item rdfs:label "%s"@en .
  ?item ?r ?item_nn .
  ?item_nn ?r2 ?item0 .
  ?item0 ?rel ?item1 .
} limit 1000
'''


def get_triplets_for_word_2_hops(word):
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    triplets = []
    for query in [query_nn_forw, query_nn2_forw]:
        data = requests.get(url, params={'query': query % word,
                                         'format': 'json'}).json()
        for item in data['results']['bindings']:
            try:
                to_item = wikidata_items.translate_from_url(item['item1']['value']) # + '|' + item['item1']['value']
                relation = wikidata_items.translate_from_url(item['rel']['value']) # + '|' + item['rel']['value']
                from_item = wikidata_items.translate_from_url(item['item0']['value'])#  + '|' + item['item0']['value']
                triplets.append((from_item, relation, to_item))
            except:
                pass
    return triplets


def get_triplets_for_word_1_hop(word):
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
    triplets = []
    for query in [query_nn_forw]:
        data = requests.get(url, params={'query': query % word,
                                         'format': 'json'}).json()
        for item in data['results']['bindings']:
            try:
                to_item = wikidata_items.translate_from_url(item['item1']['value']) # + '|' + item['item1']['value']
                relation = wikidata_items.translate_from_url(item['rel']['value']) # + '|' + item['rel']['value']
                from_item = wikidata_items.translate_from_url(item['item0']['value']) # + '|' + item['item0']['value']
                triplets.append((from_item, relation, to_item))
            except:
                pass
    return triplets


if __name__ == '__main__':
    print('Getting triplets')
    triplets = get_triplets_for_word_2_hops('Brazil')
    [print(triplet) for triplet in triplets]
    print(len(triplets))
