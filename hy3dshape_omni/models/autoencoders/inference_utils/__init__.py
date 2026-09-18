# -*- coding: utf-8 -*-
"""
Tencent is pleased to support the open source community by making Tencent Hunyuan 3D Omni available.

Copyright (C) 2025 Tencent.  All rights reserved. The below software and/or models in this 
distribution may have been modified by Tencent ("Tencent Modifications"). All Tencent Modifications 
are Copyright (C) Tencent.

Tencent Hunyuan 3D Omni is licensed under the TENCENT HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT 
except for the third-party components listed below, which is licensed under different terms. 
Tencent Hunyuan 3D Omni does not impose any additional limitations beyond what is outlined in the 
respective licenses of these third-party components. Users must comply with all terms and conditions 
of original licenses of these third-party components and must ensure that the usage of the third party 
components adheres to all relevant laws and regulations. 

For avoidance of doubts, Tencent Hunyuan 3D Omni means training code, inference-enabling code, parameters, 
and/or weights of this Model, which are made publicly available by Tencent in accordance with TENCENT 
HUNYUAN 3D OMNI COMMUNITY LICENSE AGREEMENT.
"""

from .extract_geometry_base import BaseGeometryExtractor
from .extract_geometry_block import BlockGeometryExtractor, extract_geometry_block
from .extract_geometry_vanilla import VanillaGeometryExtractor, extract_geometry_vanilla
from .extract_geometry_fast_v1 import FastGeometryExtractorV1, extract_geometry_fast_v1 as fast_extract_geometry
from .extract_geometry_fast_v2 import FastGeometryExtractorV2, extract_geometry_fast_v2 as extract_geometry_fast

extract_geometry = extract_geometry_fast
