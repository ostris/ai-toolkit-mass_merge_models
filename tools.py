import base64
import hashlib
import json
from collections import OrderedDict
from typing import Union


def get_hash_from_dict(hash_dict: Union[dict, OrderedDict]) -> str:
    hash_input = json.dumps(hash_dict, sort_keys=True).encode('utf-8')
    hash_str = base64.urlsafe_b64encode(hashlib.md5(hash_input).digest()).decode('ascii')
    hash_str = hash_str.replace('=', '')
    return hash_str