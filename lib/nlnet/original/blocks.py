from .block_v1 import Block
from .block_v2 import BlockV2
from .block_v3 import BlockV3
from .block_v4 import BlockV4
from .block_v5 import BlockV5

def get_block_version(block_version):
    if block_version == "v1":
        return Block
    elif block_version == "v2":
        return BlockV2
    elif block_version == "v3":
        return BlockV3
    elif block_version == "v4":
        return BlockV4
    elif block_version == "v5":
        return BlockV5
    else:
        raise ValueError("Uknown block type [%s]" % block_version)
