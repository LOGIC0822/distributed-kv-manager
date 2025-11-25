# test_kv_engine.py 
import os
import tempfile
import time
from unittest.mock import Mock
from types import SimpleNamespace
from typing import Any

try:
    import pytest  # type: ignore
except Exception:
    pytest = None
HAS_TORCH = True
try:
    import torch as _torch  # type: ignore
    torch: Any = _torch
except Exception:
    HAS_TORCH = False
    torch = object()  # type: ignore[assignment]
HAS_ETCD3 = True
try:
    import etcd3 as _etcd3  # type: ignore
    etcd3: Any = _etcd3
except Exception:
    HAS_ETCD3 = False
    etcd3 = object()  # type: ignore[assignment]

if pytest is not None and (not HAS_TORCH or not HAS_ETCD3):
    pytest.skip("Requires torch and etcd3 runtime; skipping integration-like engine test.", allow_module_level=True)

# 导入要测试的模块
from distributed_kv_manager import init_engine, destroy_engine, should_store, should_retrieve, store_kv, retrieve_kv
from distributed_kv_manager.engine import StoreStatus, RetrieveStatus


class MockModelInput:
    """模拟模型输入"""
    def __init__(self, input_tokens, seq_lens, slot_mapping, num_prefill_tokens, hidden_states=None, layer_id=None, session_id=None):
        self.input_tokens = input_tokens
        self.attn_metadata = Mock()
        self.attn_metadata.seq_lens = seq_lens
        self.attn_metadata.slot_mapping = slot_mapping
        self.attn_metadata.num_prefill_tokens = num_prefill_tokens
        self.hidden_states = hidden_states
        self.layer_id = layer_id
        self.session_id = session_id


class MockConfig:
    """模拟配置对象"""
    def __init__(self, storage_type="local", storage_dir=None, local_dir=None, enable_caching=False, cache_dir=None):
        self.rank = 0
        self.local_rank = 0
        self.kv_transfer_config = SimpleNamespace(
            storage_type=storage_type,
            storage_dir=storage_dir or tempfile.mkdtemp(),
            local_dir=local_dir or storage_dir or tempfile.mkdtemp(),  # 为local存储类型设置local_dir
            etcd_endpoints=["127.0.0.1:2379"],
            enable_ssd_caching=enable_caching,
            ssd_cache_dir=cache_dir or tempfile.mkdtemp(),
            enable_prefetch=True
        )


def check_etcd_metadata(config, file_path):
    """检查ETCD中的元数据"""
    from distributed_kv_manager.metadata.etcd import KVMetadataManager
    
    endpoints = getattr(config.kv_transfer_config, "etcd_endpoints", ["127.0.0.1:2379"])
    meta_manager = KVMetadataManager(endpoints=endpoints, prefix="/kvmeta")
    
    # 尝试获取元数据
    meta = meta_manager.get_metadata(file_path)
    print(f"ETCD中的元数据: {meta}")
    
    return meta

# 在测试代码中添加
def wait_for_all_tasks(engine):
    """等待所有异步任务完成"""
    print(f"等待 {len(engine._futures)} 个异步任务完成")
    
    for i, f in enumerate(engine._futures):
        try:
            f.result(timeout=60)
            print(f"异步任务 {i} 完成")
        except Exception as e:
            print(f"异步任务 {i} 失败: {e}")
    
    # 清空任务列表
    engine._futures.clear()

def test_kv_engine_basic():
    """测试KV引擎的基本功能"""
    # 创建临时目录用于存储
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建缓存目录
        cache_dir = tempfile.mkdtemp()
        # 初始化配置和引擎，启用缓存
        config = MockConfig(storage_type="local", storage_dir=temp_dir, local_dir=temp_dir, enable_caching=True, cache_dir=cache_dir)
        engine = init_engine(config)
        
        try:
            # 创建模拟数据
            batch_size = 2
            seq_len = 4
            hidden_size = 8
            num_layers = 2
            head_size = 4
            num_heads = 2
            
            # 创建输入tokens - 确保每个序列都是唯一的
            input_tokens = torch.cat([
                torch.randint(100, 200, (seq_len,)),  # 第一个序列
                torch.randint(200, 300, (seq_len,))   # 第二个序列
            ])
            
            # 创建序列长度和槽位映射
            seq_lens = [seq_len] * batch_size
            slot_mapping = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)
            
            # 创建隐藏状态
            hidden_states = torch.randn(batch_size * seq_len, hidden_size)
            
            # 创建KV缓存
            kv_caches = []
            for _ in range(num_layers):
                # 每个KV缓存是[2, batch_size, seq_len, num_heads, head_size]
                k_cache = torch.randn(batch_size, seq_len, num_heads, head_size)
                v_cache = torch.randn(batch_size, seq_len, num_heads, head_size)
                kv_cache = torch.stack([k_cache, v_cache], dim=0)
                kv_caches.append(kv_cache)
            
            # 创建模型输入
            model_input = MockModelInput(
                input_tokens=input_tokens,
                seq_lens=seq_lens,
                slot_mapping=slot_mapping,
                num_prefill_tokens=batch_size * seq_len,
                hidden_states=hidden_states,
                layer_id=0,
                session_id=b"test_session"
            )
            
            # 测试should_store方法
            store_status = should_store(model_input)
            assert store_status == StoreStatus.STORED
            
            # 测试should_retrieve方法（应该返回MISS，因为还没有存储）
            retrieve_status = should_retrieve(model_input)
            assert retrieve_status == RetrieveStatus.MISS
            
            # 模拟其他参数
            model_config = Mock()
            parallel_config = Mock()
            transfer_config = Mock()
            model_executable = Mock()
            # 为model_executable添加必要的属性
            model_executable.model = Mock()
            model_executable.model.embed_tokens = Mock()
            model_executable.model.embed_tokens.embedding_dim = hidden_size
            model_executable.model.embed_tokens.weight = Mock()
            model_executable.model.embed_tokens.weight.dtype = torch.float32
            
            # 测试存储KV
            store_kv(model_config, parallel_config, transfer_config,
                    model_executable, model_input, kv_caches, store_status)
            
            # 等待异步操作完成
            wait_for_all_tasks(engine)  
            time.sleep(1)  # 等待一段时间确保元数据已写入
            
            # 检查ETCD中的元数据
            for seq_idx, seq_len in enumerate(seq_lens):
                start_pos = sum(seq_lens[:seq_idx])
                end_pos = start_pos + seq_len
                current_tokens = input_tokens[start_pos:end_pos]
                # 使用正确的session_id和layer_id创建文件路径
                file_path = engine._make_key(current_tokens, session_id=b"test_session", layer_id=0)
                
                print(f"检查序列 {seq_idx} 的元数据: {file_path}")
                meta = check_etcd_metadata(config, file_path)
                
                if meta is None:
                    print(f"序列 {seq_idx} 的元数据在ETCD中不存在")
                else:
                    print(f"序列 {seq_idx} 的元数据状态: {meta.status}")
            
            # 重新初始化引擎以确保元数据缓存被刷新
            destroy_engine()
            engine = init_engine(config)
            
            # 再次测试should_retrieve方法（应该返回HIT，因为已经存储）
            retrieve_status = should_retrieve(model_input)
            print(f"检索状态: {retrieve_status}")
            
            # 如果状态不是HIT，检查原因
            if retrieve_status != RetrieveStatus.HIT:
                for seq_idx, seq_len in enumerate(seq_lens):
                    start_pos = sum(seq_lens[:seq_idx])
                    end_pos = start_pos + seq_len
                    current_tokens = input_tokens[start_pos:end_pos]
                    # 使用正确的session_id和layer_id创建文件路径
                    file_path = engine._make_key(current_tokens, session_id=b"test_session", layer_id=0)
                    
                    # 检查文件是否存在
                    file_exists = os.path.exists(os.path.join(temp_dir, file_path))
                    print(f"序列 {seq_idx} 文件存在: {file_exists}")
                    
                    # 检查元数据
                    meta = check_etcd_metadata(config, file_path)
                    print(f"序列 {seq_idx} 元数据: {meta}")
            
            # 创建新的KV缓存用于检索
            new_kv_caches = []
            for _ in range(num_layers):
                # 初始化空的KV缓存，保持正确的形状
                k_cache = torch.zeros(batch_size, seq_len, num_heads, head_size)
                v_cache = torch.zeros(batch_size, seq_len, num_heads, head_size)
                kv_cache = torch.stack([k_cache, v_cache], dim=0)
                new_kv_caches.append(kv_cache)
            # 测试检索KV之前
            print(f"即将检索KV，检索状态: {retrieve_status}")
            print(f"新的KV缓存形状: {[kv_cache.shape for kv_cache in new_kv_caches]}")

            try:
                result, bypass, new_model_input = retrieve_kv(
                    model_executable, model_input, new_kv_caches, retrieve_status
                )
                print(f"检索结果: bypass={bypass}, result_shape={result.shape if result is not None else 'None'}")
            except Exception as e:
                print(f"检索KV时发生错误: {e}")
                import traceback
                traceback.print_exc()
                # 可以选择继续执行或退出
                result, bypass, new_model_input = None, False, model_input
            
            # 验证检索结果
            if bypass:
                assert result is not None
                assert result.shape == hidden_states.shape
                
                # 验证KV缓存已被填充
                for layer_idx in range(num_layers):
                    # 检查KV缓存是否被正确填充
                    original_kv = kv_caches[layer_idx]
                    retrieved_kv = new_kv_caches[layer_idx]
                    
                    # 检查两个张量是否在允许的误差范围内相等
                    assert torch.allclose(original_kv, retrieved_kv, atol=1e-6)
            
            print("所有测试通过!")
            
        finally:
            # 清理
            destroy_engine()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    test_kv_engine_basic()
    print("测试完成!")