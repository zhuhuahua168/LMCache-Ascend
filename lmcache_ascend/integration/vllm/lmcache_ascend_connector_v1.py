# SPDX-License-Identifier: Apache-2.0
# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger

# First Party
from lmcache_ascend import _build_info

if _build_info.__framework_name__ == "pytorch":
    # First Party
    import lmcache_ascend  # noqa: F401
elif _build_info.__framework_name__ == "mindspore":
    # First Party
    import lmcache_ascend.mindspore  # noqa: F401
else:
    raise ValueError("Unsupported Framework")

# Third Party
from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)

    def save_kv_layer(self, layer_name, kv_cache_layer, attn_metadata):
        if self._lmcache_engine.lmcache_engine is None:
            logger.debug(
                "save_kv_layer called but lmcache_engine is not initialized yet, "
                "skipping (likely during warm-up). layer_name=%s",
                layer_name,
            )
            return
        self._lmcache_engine.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)

    def load_kv_layer(self, layer_name, kv_cache_layer, attn_metadata):
        if self._lmcache_engine.lmcache_engine is None:
            logger.debug(
                "load_kv_layer called but lmcache_engine is not initialized yet, "
                "skipping (likely during warm-up). layer_name=%s",
                layer_name,
            )
            return
        self._lmcache_engine.load_kv_layer(layer_name, kv_cache_layer, attn_metadata)
