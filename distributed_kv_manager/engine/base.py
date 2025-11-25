from enum import Enum

class StoreStatus(Enum):
    STORED = 1
    SKIPPED = 2

class RetrieveStatus(Enum):
    HIT = 1
    MISS = 2

class DistributedKVEngineBase:
    """KV缓存存储后端抽象"""
    def should_store(self, model_input) -> StoreStatus:
        raise NotImplementedError

    def store_kv(self, model_config, parallel_config, transfer_config,
                 model_executable, model_input, kv_caches, store_status,
                 hidden_or_intermediate_states=None):
        raise NotImplementedError

    def should_retrieve(self, model_input) -> RetrieveStatus:
        raise NotImplementedError

    def retrieve_kv(self, model_executable, model_input, kv_caches, retrieve_status):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
