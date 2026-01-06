import os
import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Dict, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import cv2
from metai.utils import get_config
from metai.utils import MetConfig, MetLabel, MetRadar, MetNwp, MetGis, MetVar

# 默认通道列表常量
# _DEFAULT_CHANNELS: List[Union[MetLabel, MetRadar, MetNwp, MetGis]] = [
#     MetLabel.RA, MetRadar.CR, MetRadar.CAP20, MetRadar.CAP30, MetRadar.CAP40, MetRadar.CAP50, MetRadar.CAP60, MetRadar.CAP70, MetRadar.ET, MetRadar.HBR, MetRadar.VIL,
#     MetNwp.WS925, MetNwp.WS700, MetNwp.WS500, MetNwp.Q1000, MetNwp.Q850, MetNwp.Q700, MetNwp.PWAT, MetNwp.PE, MetNwp.TdSfc850, MetNwp.LCL, MetNwp.KI, MetNwp.CAPE,
#     MetGis.LAT, MetGis.LON, MetGis.DEM, MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS
# ]

_DEFAULT_CHANNELS: List[Union[MetLabel, MetRadar, MetNwp, MetGis]] = [
    MetLabel.RA, MetRadar.CR, 
    MetGis.LAT, MetGis.LON, MetGis.DEM, MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS
]

@dataclass
class MetSample:
    """
    天气样本数据结构，基于竞赛数据目录结构设计。
    用于管理单个天气样本的元数据和配置信息。
    """
    sample_id: str
    timestamps: List[str]
    met_config: MetConfig
    is_debug: bool = field(default_factory=lambda: False)
    is_train: bool = field(default_factory=lambda: True)
    test_set: str = field(default_factory=lambda: "TestSet")
    
    channels: List[Union[MetLabel, MetRadar, MetNwp, MetGis]] = field(
        default_factory=lambda: _DEFAULT_CHANNELS.copy()
    )
    channel_size: int = field(default_factory=lambda: len(_DEFAULT_CHANNELS))
    
    task_mode: str = field(default_factory=lambda: 'precipitation')
    default_shape: Tuple[int, int] = field(default_factory=lambda: (301, 301))
    max_timesteps: int = field(default_factory=lambda: 10)
    out_timesteps: int = field(default_factory=lambda: 20)

    _gis_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, init=False, repr=False)
    _sample_id_parts: Optional[List[str]] = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, sample_id: str, timestamps: List[str], config: Optional['MetConfig'] = None, **kwargs) -> 'MetSample':
        if config is None:
            config = get_config(is_debug=kwargs.get('is_debug', False))
        
        if 'is_debug' in kwargs:
            config.is_debug = kwargs['is_debug']
        else:
            kwargs['is_debug'] = config.is_debug

        return cls(
            sample_id=sample_id,
            timestamps=timestamps,
            met_config=config,
            **kwargs
        )
    
    def _get_sample_id_parts(self) -> List[str]:
        if self._sample_id_parts is None:
            self._sample_id_parts = self.sample_id.split('_')
        return self._sample_id_parts
    
    @cached_property
    def task_id(self) -> str:
        return self._get_sample_id_parts()[0]

    @cached_property
    def region_id(self) -> str:
        return self._get_sample_id_parts()[1]

    @cached_property
    def time_id(self) -> str:
        return self._get_sample_id_parts()[2]

    @cached_property
    def station_id(self) -> str:
        return self._get_sample_id_parts()[3]

    @cached_property
    def radar_type(self) -> str:
        return self._get_sample_id_parts()[4]

    @cached_property
    def batch_id(self) -> str:
        return self._get_sample_id_parts()[5]

    @cached_property
    def case_id(self) -> str:
        parts = self._get_sample_id_parts()
        return '_'.join(parts[:4])

    @cached_property
    def base_path(self) -> str:
        return os.path.join(
            self.met_config.root_path,
            self.task_id,
            "TrainSet" if self.is_train else self.test_set,
            self.region_id,
            self.case_id,
        )
    
    @property
    def root_path(self) -> str:
        return self.met_config.root_path
    
    @property
    def gis_data_path(self) -> str:
        return self.met_config.gis_data_path
    
    @property
    def file_date_format(self) -> str:
        return self.met_config.file_date_format
    
    @property
    def nwp_prefix(self) -> str:
        return self.met_config.nwp_prefix
    
    @property
    def user_id(self) -> str:
        return self.met_config.user_id
    
    @property
    def metadata(self) -> Dict:
        metadata_dict = vars(self).copy()
        metadata_dict.pop('_gis_cache', None)
        return metadata_dict

    def str_to_datetime(self, time_str: str) -> datetime:
        return datetime.strptime(time_str, self.file_date_format)

    def datetime_to_str(self, datetime_obj: datetime) -> str:
        return datetime_obj.strftime(self.file_date_format)

    def _ensure_shape(self, data: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """[新增] 强制确保数据尺寸符合 default_shape (301, 301)"""
        # [Fix] 增加对空数组的防御性检查
        if data is None or data.size == 0 or 0 in data.shape:
             return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)

        if data.shape != self.default_shape:
            # cv2.resize expects (width, height), which is (W, H)
            # self.default_shape is (H, W) -> (301, 301)
            dsize = (self.default_shape[1], self.default_shape[0])
            try:
                data = cv2.resize(data, dsize, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask.astype(np.uint8), dsize, interpolation=cv2.INTER_NEAREST).astype(bool)
            except cv2.error:
                # 如果 resize 仍然失败，回退到全0
                return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)
        return data, mask

    def normalize(self, file_path: str, min_value: float = 0, max_value: float = 300) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            data = np.load(file_path)
            # [Fix] 检查空数组，如果为空则视为无效数据，返回 None
            if data.size == 0 or 0 in data.shape:
                return None, None
            
            valid_mask = np.isfinite(data)
            data = np.nan_to_num(data, nan=min_value, neginf=min_value, posinf=max_value)
            inv_denom = 1.0 / (max_value - min_value)
            scale = (data - min_value) * inv_denom
            np.clip(scale, 0.0, 1.0, out=scale)
            return scale.astype(np.float32, copy=False), valid_mask.astype(bool, copy=False)
        except Exception:
            return None, None

    def normalize_with_mask(self, file_path: str, min_value: float = 0, max_value: float = 300, missing_value: float = -9) -> tuple[np.ndarray, np.ndarray]:
        try:
            data = np.load(file_path)
            # [Fix] 检查空数组
            if data.size == 0 or 0 in data.shape:
                default_data = np.zeros(self.default_shape, dtype=np.float32)
                default_mask = np.zeros(self.default_shape, dtype=bool)
                return default_data, default_mask

            valid_mask = (data != missing_value) & np.isfinite(data)
            data = np.nan_to_num(data, nan=min_value, neginf=min_value, posinf=max_value)
            inv_denom = 1.0 / (max_value - min_value)
            scale = (data - min_value) * inv_denom
            np.clip(scale, 0.0, 1.0, out=scale)
            return scale.astype(np.float32, copy=False), valid_mask.astype(bool, copy=False)
        except Exception:
            default_data = np.zeros(self.default_shape, dtype=np.float32)
            default_mask = np.zeros(self.default_shape, dtype=bool)
            return default_data, default_mask
    
    def load_input_data(self) -> Tuple[np.ndarray, np.ndarray]:
        past_timesteps = self.timestamps[:self.max_timesteps] 
        future_start = self.max_timesteps
        future_end = future_start + self.out_timesteps
        future_timesteps = self.timestamps[future_start:future_end]
        
        self._preload_gis_data()

        # --- A. 加载基础输入 ---
        past_series = []
        past_masks = []
        
        for i, timestamp in enumerate(past_timesteps):
            frames = []
            masks = []
            for channel in self.channels:
                data, mask = self._load_channel_frame_with_fallback(channel, timestamp, i, self.timestamps)
                frames.append(data)
                masks.append(mask)
            past_series.append(np.stack(frames, axis=0))
            past_masks.append(np.stack(masks, axis=0))
            
        input_base = np.stack(past_series, axis=0)
        mask_base = np.stack(past_masks, axis=0)

        # --- B. 加载未来 NWP 引导 ---
        nwp_channels = [c for c in self.channels if isinstance(c, MetNwp)]
        
        if len(nwp_channels) > 0 and len(future_timesteps) == self.out_timesteps:
            future_nwp_series = []
            future_nwp_masks = []
            
            for i, timestamp in enumerate(future_timesteps):
                frames = []
                masks = []
                for channel in nwp_channels:
                    global_idx = future_start + i
                    data, mask = self._load_channel_frame_with_fallback(channel, timestamp, global_idx, self.timestamps)
                    frames.append(data)
                    masks.append(mask)
                future_nwp_series.append(np.stack(frames, axis=0))
                future_nwp_masks.append(np.stack(masks, axis=0))
            
            input_future_nwp = np.stack(future_nwp_series, axis=0)
            mask_future_nwp = np.stack(future_nwp_masks, axis=0)
            
            B_t, C_n, H_t, W_t = input_future_nwp.shape
            fold_factor = self.out_timesteps // self.max_timesteps
            
            if self.out_timesteps % self.max_timesteps == 0:
                input_folded = input_future_nwp.reshape(self.max_timesteps, fold_factor * C_n, H_t, W_t)
                mask_folded = mask_future_nwp.reshape(self.max_timesteps, fold_factor * C_n, H_t, W_t)
                input_data = np.concatenate([input_base, input_folded], axis=1)
                input_mask = np.concatenate([mask_base, mask_folded], axis=1)
            else:
                input_data = input_base
                input_mask = mask_base
        else:
            input_data = input_base
            input_mask = mask_base

        return input_data, input_mask

    def load_target_data(self) -> tuple[np.ndarray, np.ndarray]:
        target_data = []
        valid_mask = []

        start_idx = self.max_timesteps
        end_idx = start_idx + self.out_timesteps
        target_timestamps = self.timestamps[start_idx : end_idx]

        for timestamp in target_timestamps:
            file_path = os.path.join(
                self.base_path,
                MetVar.LABEL.name,
                MetLabel.RA.name,
                f"CP_Label_{MetLabel.RA.name}_{self.station_id}_{timestamp}.npy"
            )

            min_val, max_val = self._get_channel_limits(MetLabel.RA)
            missing_val = getattr(MetLabel.RA, 'missing_value', -9.0)
            data, mask = self.normalize_with_mask(file_path, min_val, max_val, missing_value=missing_val)
            
            if data is not None and mask is not None:
                data, mask = self._ensure_shape(data, mask)

            target_data.append(data)
            valid_mask.append(mask)

        target_data = np.expand_dims(np.stack(target_data, axis=0), axis=1)
        valid_mask = np.expand_dims(np.stack(valid_mask, axis=0), axis=1)
        return target_data, valid_mask

    def to_numpy(self, is_train: bool=False) -> Tuple[Dict, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        input_data, input_mask = self.load_input_data()
        
        if is_train:
            target_data, target_mask = self.load_target_data()
            return self.metadata, input_data, target_data, target_mask, input_mask
        else:
            return self.metadata, input_data, None, None, input_mask
    
    def _get_channel_limits(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis]) -> tuple[float, float]:
        return (
            float(getattr(channel, "min", 0.0)),
            float(getattr(channel, "max", 1.0))
        )

    def _load_label_frame(self, var: MetLabel, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        file_path = os.path.join(
            self.base_path, var.parent, var.value, 
            f"{self.task_id}_Label_{var.value}_{self.station_id}_{timestamp}.npy"
        )
        min_val, max_val = self._get_channel_limits(var)
        data, mask = self.normalize(file_path, min_val, max_val)
        
        if data is not None and mask is not None:
            data, mask = self._ensure_shape(data, mask)
        return data, mask

    def _load_radar_frame(self, var: MetRadar, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        file_path = os.path.join(
            self.base_path, var.parent, var.value, 
            f"{self.task_id}_RADA_{self.station_id}_{timestamp}_{self.radar_type}_{var.value}.npy"
        )
        min_val, max_val = self._get_channel_limits(var)
        data, mask = self.normalize(file_path, min_val, max_val)
        
        if data is not None and mask is not None:
            data, mask = self._ensure_shape(data, mask)
        return data, mask

    def _load_nwp_frame(self, var: MetNwp, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]: # type: ignore
        obs_time = self.str_to_datetime(timestamp)

        if obs_time.minute < 30:
            obs_time = obs_time.replace(minute=0)
        else:
            obs_time = obs_time.replace(minute=0) + timedelta(hours=1)

        file_path = os.path.join(
            self.base_path, var.parent, var.name, 
            f"{self.task_id}_{self.nwp_prefix}_{self.station_id}_{self.datetime_to_str(obs_time)}_{var.value}.npy"
        )
        min_val, max_val = self._get_channel_limits(var)
        data, mask = self.normalize(file_path, min_val, max_val)
        
        if data is None or mask is None:
            return None, None
            
        dsize = (self.default_shape[1], self.default_shape[0])
        try:
            resized_data = cv2.resize(data, dsize, interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(mask.astype(np.uint8), dsize, interpolation=cv2.INTER_NEAREST).astype(bool)
        except cv2.error:
            # Fallback
            return None, None
        
        return resized_data.astype(np.float32, copy=False), resized_mask

    def _load_gis_frame(self, var: MetGis, timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        full_mask = np.full(self.default_shape, True, dtype=bool)
        
        if var in [MetGis.MONTH_SIN, MetGis.MONTH_COS, MetGis.HOUR_SIN, MetGis.HOUR_COS]:
            try:
                obs_time = self.str_to_datetime(timestamp)
                raw_val = 0.0
                if var in [MetGis.MONTH_SIN, MetGis.MONTH_COS]:
                    angle = 2 * math.pi * float(obs_time.month) / 12.0
                    raw_val = math.sin(angle) if var == MetGis.MONTH_SIN else math.cos(angle)
                elif var in [MetGis.HOUR_SIN, MetGis.HOUR_COS]:
                    angle = 2 * math.pi * float(obs_time.hour) / 24.0
                    raw_val = math.sin(angle) if var == MetGis.HOUR_SIN else math.cos(angle)
                
                min_val, max_val = self._get_channel_limits(var)
                if max_val - min_val > 0:
                    normalized_value = (raw_val - min_val) / (max_val - min_val)
                else:
                    normalized_value = 0.5 
                
                normalized_value = max(0.0, min(1.0, normalized_value))
                data = np.full(self.default_shape, normalized_value, dtype=np.float32)
                return data, full_mask
            except Exception:
                return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)

        # 静态 GIS 文件
        try:
            file_name = f"{var.value}.npy"
            file_path = os.path.join(self.gis_data_path, self.station_id, file_name)
            min_val, max_val = self._get_channel_limits(var)
            data, mask = self.normalize(file_path, min_val, max_val)
            
            if data is not None and mask is not None:
                data, mask = self._ensure_shape(data, mask)

            if data is None:
                return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)
            return data, mask

        except Exception:
            return np.zeros(self.default_shape, dtype=np.float32), np.zeros(self.default_shape, dtype=bool)

    def _preload_gis_data(self):
        self._gis_cache.clear()
        default_data = np.zeros(self.default_shape, dtype=np.float32)
        default_mask = np.zeros(self.default_shape, dtype=bool)
        
        for channel in self.channels:
            if isinstance(channel, MetGis):
                cache_key = f"gis_{channel.value}"
                if cache_key not in self._gis_cache:
                    data, mask = self._load_gis_frame(channel, self.timestamps[0])
                    self._gis_cache[cache_key] = (data if data is not None else default_data, 
                                                 mask if mask is not None else default_mask)

    def _load_channel_frame(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis], timestamp: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        loader_map = {
            MetLabel: self._load_label_frame,
            MetRadar: self._load_radar_frame,
            MetGis: self._load_gis_frame,
            MetNwp: self._load_nwp_frame,
        }

        loader = None
        for enum_cls, handler in loader_map.items():
            if isinstance(channel, enum_cls):
                loader = handler
                break

        if loader is None:
            return None, None

        data, mask = loader(channel, timestamp)

        if data is None:
            return None, None

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask, dtype=bool)

        return data.astype(np.float32, copy=False), mask.astype(bool, copy=False)

    def _load_temporal_data_with_fallback(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis], timestamp: str, 
                                         timestep_idx: int, all_timestamps: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        default_data = np.zeros(self.default_shape, dtype=np.float32)
        default_mask = np.zeros(self.default_shape, dtype=bool)

        data, mask = self._load_channel_frame(channel, timestamp)
        if data is not None and mask is not None:
            return data, mask
        
        if timestep_idx > 0:
            prev_timestamp = all_timestamps[timestep_idx - 1]
            data, mask = self._load_channel_frame(channel, prev_timestamp)
            if data is not None:
                return data, default_mask
        
        if timestep_idx < len(all_timestamps) - 1:
            next_timestamp = all_timestamps[timestep_idx + 1]
            data, mask = self._load_channel_frame(channel, next_timestamp)
            if data is not None:
                return data, default_mask
        
        return default_data, default_mask

    def _load_channel_frame_with_fallback(self, channel: Union[MetLabel, MetRadar, MetNwp, MetGis], timestamp: str, 
                                         timestep_idx: int, all_timestamps: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        default_data = np.zeros(self.default_shape, dtype=np.float32)
        default_mask = np.zeros(self.default_shape, dtype=bool)

        if isinstance(channel, MetGis):
            cache_key = f"gis_{channel.value}"
            if cache_key in self._gis_cache:
                return self._gis_cache[cache_key]
            data, mask = self._load_gis_frame(channel, timestamp)
            if data is not None and mask is not None:
                self._gis_cache[cache_key] = (data, mask)
                return data, mask
            return default_data, default_mask
        
        if isinstance(channel, (MetRadar, MetNwp)):
            return self._load_temporal_data_with_fallback(channel, timestamp, timestep_idx, all_timestamps)
        
        data, mask = self._load_channel_frame(channel, timestamp)
        if data is not None and mask is not None:
            return data, mask
        
        return default_data, default_mask