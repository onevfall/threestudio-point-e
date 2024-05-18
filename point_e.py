
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config

from dataclasses import dataclass, field
from typing import List
import os

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *

from tqdm.auto import tqdm

@threestudio.register("point-e-guidance")
class point_e_Guidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):

        skip: int = 4
        cache_dir: str = "custom/threestudio-point-e/point-e/cache"

    cfg: Config

    def configure(self) -> None:
        pass
    
    def densify(self, factor=2):
        pass

    def __call__(
        self,
        prompt,
        **kwargs,
    ):
        
        threestudio.info(f"Loading point-e guidance ...")
        
        device = self.device 
        print('creating base model...')
        base_name = 'base40M-textvec'
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

        print('creating upsample model...')
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

        print('downloading base checkpoint...')
        base_model.load_state_dict(load_checkpoint(base_name, device, cache_dir=self.cfg.cache_dir))

        print('downloading upsampler checkpoint...')
        upsampler_model.load_state_dict(load_checkpoint('upsample', device, cache_dir=self.cfg.cache_dir))

        threestudio.info(f"Loaded point-e guidance!")
        
        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
        )
        
        # Produce a sample from the model.
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x
            
        pc = sampler.output_to_point_clouds(samples)[0]
        
        skip = self.cfg.skip
        
        rgb = np.stack(
                    [pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=-1
            )
        coords = pc.coords
            
        coords = coords[::skip]
        rgb = rgb[::skip]

        return coords,rgb
