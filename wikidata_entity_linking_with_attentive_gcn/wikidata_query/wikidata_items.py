import os
import logging

_path = os.path.dirname(__file__)


class WikidataItems:
    _filename = os.path.join(_path, '../data/wikidata_items.csv')
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self._logger.warning('Loading items')
        self._items_dict = {}
        self._reverse_dict = {}
        with open(self._filename, encoding='utf8') as f:
            for item in f.readlines():
                item = item[:-1]
                item_key, item_value = item.split('\t')[:2]
                if ':' in item_value or len(item_value) < 2:
                    continue
                if item_key not in self._items_dict:
                    self._items_dict[item_key] = item_value
                try:
                    self._reverse_dict[item_value.lower()].append(item_key)
                except:
                    self._reverse_dict[item_value.lower()] = [item_key]
                # add also string without '.'
                try:
                    self._reverse_dict[item_value.lower().replace('.', '')].append(item_key)
                except:
                    self._reverse_dict[item_value.lower().replace('.', '')] = [item_key]

        self._logger.warning('Items loaded')

    def __getitem__(self, item):
        return self._items_dict[item]

    def translate_from_url(self, url):
        if '/' in url and '-' not in url:
            item = url.split('/')[-1]
        elif '/' in url and '-' in url:
            item = url.split('/')[-1].split('-')[0]
        else:
            item = url
        return self[item]

    def reverse_lookup(self, word):
        return self._reverse_dict[word.lower()]


wikidata_items = WikidataItems()
