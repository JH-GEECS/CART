import os
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

### merge definitions ###
MERGE_TYPE_LIST = ["static", "adaptive", "indexing"]

MERGE_METHOD_STATIC_LIST = ["CART", "TA", "AVG"]
MERGE_METHOD_INDEXING_LIST = ["CART-INDEXING"]

MERGE_METHOD_LIST = MERGE_METHOD_STATIC_LIST + MERGE_METHOD_INDEXING_LIST
