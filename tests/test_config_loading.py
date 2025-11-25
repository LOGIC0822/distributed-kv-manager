try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
from distributed_kv_manager import init_engine, destroy_engine, load_config_from_json
from distributed_kv_manager.storage.factory import StorageFactory

def test_config_loading():
    """测试从config.json加载配置"""
    print("=== 测试从config.json加载配置 ===")
    
    # 从config.json加载配置
    config = load_config_from_json()
    print(f"加载的配置: {config}")
    print(f"存储类型: {getattr(config.kv_transfer_config, 'storage_type', 'N/A')}")
    print(f"存储目录: {getattr(config.kv_transfer_config, 'storage_dir', 'N/A')}")
    print(f"ETCD端点: {getattr(config.kv_transfer_config, 'etcd_endpoints', 'N/A')}")
    print(f"Crail目录: {getattr(config.kv_transfer_config, 'crail_dir', 'N/A')}")
    print(f"本地目录: {getattr(config.kv_transfer_config, 'local_dir', 'N/A')}")
    
    # 测试存储工厂
    print("\n=== 测试存储工厂 ===")
    storage = StorageFactory.create_storage(config)
    assert storage is not None
    test_data = b"test data for storage"
    test_file = "test_config_file.txt"
    assert storage.upload(test_file, test_data)
    assert storage.exists(test_file)
    downloaded_data = storage.download(test_file)
    assert downloaded_data == test_data
    if hasattr(storage, 'delete'):
        storage.delete(test_file)
    
    # 初始化引擎（若缺少 torch，跳过此段）
    if torch is None:
        return

    print("\n=== 测试引擎初始化 ===")
    engine = init_engine(config)
    try:
        print(f"引擎初始化成功: {type(engine).__name__}")
        print(f"引擎存储实例类型: {type(engine._storage).__name__}")

        # 构造测试数据
        seq_len = 3
        num_layers = 2
        num_heads = 2
        head_size = 4

        input_tokens = torch.arange(seq_len)
        kv_caches = [torch.randn(2, seq_len, num_heads, head_size) for _ in range(num_layers)]
        hidden_states = torch.randn(seq_len, 8)

        class ModelInput:
            def __init__(self, input_tokens, hidden_states):
                self.input_tokens = input_tokens
                self.hidden_states = hidden_states
                self.attn_metadata = type('AttnMetadata', (), {
                    'seq_lens': [seq_len],
                    'slot_mapping': torch.arange(seq_len).unsqueeze(0),
                    'num_prefill_tokens': seq_len
                })()

        model_input = ModelInput(input_tokens, hidden_states)
        store_status = engine.should_store(model_input)
        assert store_status is not None
    finally:
        destroy_engine()
        print("引擎已销毁")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_config_loading()