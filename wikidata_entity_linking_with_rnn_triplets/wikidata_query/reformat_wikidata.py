import ijson

_json_file_name = '../data/wikidata-20170904-all.json'
_items_file = '../data/wikidata_strings.csv'

_outfile = open(_items_file, 'w')

item_id = None
item_value = None

for prefix, the_type, value in ijson.parse(open(_json_file_name)):
    try:
        value = value.encode("utf-8", "surrogateescape")
    except:
        value = None
    if prefix == 'item.id' and the_type == 'string':
        item_id = value
    if prefix == 'item.labels.en.value' and the_type == 'string':
        item_value = value
    if item_id and item_value:
        _outfile.write(item_id.decode('utf8') + '\t' + item_value.decode('utf8') + '\n')
        item_id = None
        item_value = None