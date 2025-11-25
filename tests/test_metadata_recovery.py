# test_metadata_recovery.py 
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
    torch = object()

HAS_ETCD3 = True
try:
    import etcd3 as _etcd3  # type: ignore
    etcd3: Any = _etcd3
except Exception:
    HAS_ETCD3 = False
    etcd3 = object()

if pytest is not None and (not HAS_TORCH or not HAS_ETCD3):
    pytest.skip("Requires torch and etcd3; skipping metadata recovery tests.", allow_module_level=True)

# 导入要测试的模块
from distributed_kv_manager import init_engine, destroy_engine, should_store, should_retrieve, store_kv, retrieve_kv
from distributed_kv_manager.engine import StoreStatus, RetrieveStatus
from distributed_kv_manager.metadata.etcd import KVMetadataManager


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


def test_metadata_recovery_from_storage():
    """测试从存储中恢复元数据功能"""
    print("开始测试从存储中恢复元数据功能...")
    
    # 创建临时目录用于存储
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建缓存目录
        cache_dir = tempfile.mkdtemp()
        # 初始化配置和引擎，启用缓存
        config = MockConfig(storage_type="local", storage_dir=temp_dir, local_dir=temp_dir, enable_caching=True, cache_dir=cache_dir)
        engine = init_engine(config)
        
        try:
            # 创建模拟数据
            batch_size = 1
            seq_len = 4
            hidden_size = 8
            num_layers = 2
            head_size = 4
            num_heads = 2
            
            # 创建输入tokens
            input_tokens = torch.randint(100, 200, (seq_len,))
            
            # 创建序列长度和槽位映射
            seq_lens = [seq_len]
            slot_mapping = torch.arange(seq_len).reshape(1, seq_len)
            
            # 创建隐藏状态
            hidden_states = torch.randn(seq_len, hidden_size)
            
            # 创建KV缓存（包含 batch 维度）
            kv_caches = []
            batch_size = 1
            for _ in range(num_layers):
                k_cache = torch.randn(batch_size, seq_len, num_heads, head_size)
                v_cache = torch.randn(batch_size, seq_len, num_heads, head_size)
                kv_cache = torch.stack([k_cache, v_cache], dim=0)  # [2, B, S, H, D]
                kv_caches.append(kv_cache)
            
            # 创建模型输入
            model_input = MockModelInput(
                input_tokens=input_tokens,
                seq_lens=seq_lens,
                slot_mapping=slot_mapping,
                num_prefill_tokens=seq_len,
                hidden_states=hidden_states,
                layer_id=0,
                session_id=b"test_session_recovery"
            )
            
            # 模拟其他参数
            model_config = Mock()
            parallel_config = Mock()
            transfer_config = Mock()
            model_executable = Mock()
            store_status = StoreStatus.STORED
            
            # 存储KV数据
            print("存储KV数据...")
            store_kv(model_config, parallel_config, transfer_config,
                    model_executable, model_input, kv_caches, store_status)
            
            # 等待异步操作完成
            wait_for_all_tasks(engine)
            time.sleep(1)  # 等待一段时间确保元数据已写入
            
            # 获取文件路径
            file_path = engine._make_key(input_tokens, session_id=b"test_session_recovery", layer_id=0)
            full_file_path = os.path.join(temp_dir, file_path)
            
            print(f"存储的文件路径: {full_file_path}")
            
            # 验证文件存在
            assert os.path.exists(full_file_path), "KV数据文件应该存在"
            print("KV数据文件存在 - 验证通过")
            
            # 销毁引擎并清除元数据缓存
            destroy_engine()
            
            # 使用KVMetadataManager的recover_metadata_from_storage方法从存储中恢复元数据
            endpoints = getattr(config.kv_transfer_config, "etcd_endpoints", ["127.0.0.1:2379"])
            meta_manager = KVMetadataManager(endpoints=endpoints, prefix="/kvmeta")
            
            # 从存储中恢复元数据
            print(f"从存储路径 {temp_dir} 中恢复元数据...")
            recovered_metadata = meta_manager.recover_metadata_from_storage(temp_dir)
            
            print(f"恢复的元数据数量: {len(recovered_metadata)}")
            for path, metadata in recovered_metadata.items():
                print(f"  文件路径: {path}")
                print(f"  元数据: {metadata}")
                
            # 验证恢复的元数据
            assert len(recovered_metadata) > 0, "应该能够恢复至少一个元数据"
            
            # 将恢复的元数据写入etcd
            print("将恢复的元数据写入etcd...")
            meta_manager.write_metadata_to_etcd(recovered_metadata)
            
            print("从存储中恢复元数据测试通过!")
            
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 清理
            destroy_engine()


def test_metadata_recovery():
    """测试元数据恢复功能"""
    print("开始测试元数据恢复功能...")
    
    # 创建临时目录用于存储
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建缓存目录
        cache_dir = tempfile.mkdtemp()
        # 初始化配置和引擎，启用缓存
        config = MockConfig(storage_type="local", storage_dir=temp_dir, local_dir=temp_dir, enable_caching=True, cache_dir=cache_dir)
        engine = init_engine(config)
        
        try:
            # 创建模拟数据
            batch_size = 1
            seq_len = 4
            hidden_size = 8
            num_layers = 2
            head_size = 4
            num_heads = 2
            
            # 创建输入tokens
            input_tokens = torch.randint(100, 200, (seq_len,))
            
            # 创建序列长度和槽位映射
            seq_lens = [seq_len]
            slot_mapping = torch.arange(seq_len).reshape(1, seq_len)
            
            # 创建隐藏状态
            hidden_states = torch.randn(seq_len, hidden_size)
            
            # 创建KV缓存
            kv_caches = []
            for _ in range(num_layers):
                k_cache = torch.randn(seq_len, num_heads, head_size)
                v_cache = torch.randn(seq_len, num_heads, head_size)
                kv_cache = torch.stack([k_cache, v_cache], dim=0)
                kv_caches.append(kv_cache)
            
            # 创建模型输入
            model_input = MockModelInput(
                input_tokens=input_tokens,
                seq_lens=seq_lens,
                slot_mapping=slot_mapping,
                num_prefill_tokens=seq_len,
                hidden_states=hidden_states,
                layer_id=0,
                session_id=b"test_session_recovery"
            )
            
            # 模拟其他参数
            model_config = Mock()
            parallel_config = Mock()
            transfer_config = Mock()
            model_executable = Mock()
            store_status = StoreStatus.STORED
            
            # 存储KV数据
            print("存储KV数据...")
            store_kv(model_config, parallel_config, transfer_config,
                    model_executable, model_input, kv_caches, store_status)
            
            # 等待异步操作完成
            wait_for_all_tasks(engine)
            time.sleep(1)  # 等待一段时间确保元数据已写入
            
            # 获取文件路径
            file_path = engine._make_key(input_tokens, session_id=b"test_session_recovery", layer_id=0)
            full_file_path = os.path.join(temp_dir, file_path)
            
            print(f"存储的文件路径: {full_file_path}")
            
            # 验证文件存在
            assert os.path.exists(full_file_path), "KV数据文件应该存在"
            print("KV数据文件存在 - 验证通过")
            
            # 获取原始元数据
            original_meta = engine._meta_cache.get_metadata(key=file_path)
            assert original_meta is not None, "原始元数据应该存在"
            print(f"原始元数据: {original_meta}")
            
            # 销毁引擎并清除元数据缓存
            destroy_engine()
            
            # 直接从ETCD中删除元数据来模拟元数据丢失
            endpoints = getattr(config.kv_transfer_config, "etcd_endpoints", ["127.0.0.1:2379"])
            from distributed_kv_manager.metadata.etcd import KVMetadataManager
            meta_manager = KVMetadataManager(endpoints=endpoints, prefix="/kvmeta")
            etcd_key = f"/kvmeta/{file_path}"
            meta_manager.delete_metadata(etcd_key)
            print(f"已删除ETCD中的元数据: {etcd_key}")
            
            # 重新初始化引擎
            engine = init_engine(config)
            
            # 尝试检索KV数据（此时应该能够从文件中恢复元数据）
            retrieve_status = should_retrieve(model_input)
            print(f"元数据丢失后的检索状态: {retrieve_status}")
            
            # 创建新的KV缓存用于检索
            new_kv_caches = []
            for _ in range(num_layers):
                k_cache = torch.zeros(1, seq_len, num_heads, head_size)
                v_cache = torch.zeros(1, seq_len, num_heads, head_size)
                kv_cache = torch.stack([k_cache, v_cache], dim=0)
                new_kv_caches.append(kv_cache)
            
            # 尝试检索KV
            print("尝试检索KV数据...")
            result, bypass, new_model_input = retrieve_kv(
                model_executable, model_input, new_kv_caches, retrieve_status
            )
            
            print(f"元数据恢复后的检索结果: bypass={bypass}")
            
            # 验证是否成功检索（应该能够从文件中恢复元数据并成功检索）
            assert bypass, "应该能够成功检索KV数据"
            assert result is not None, "应该返回隐藏状态"
            
            print("元数据恢复测试通过!")
            
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 清理
            destroy_engine()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # 运行两个测试
    test_metadata_recovery_from_storage()
    test_metadata_recovery()
    print("所有元数据恢复测试完成!")