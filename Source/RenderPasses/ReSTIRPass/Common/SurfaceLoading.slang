/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.RaytracingInline;
import Utils.Geometry.GeometryHelpers;

__exported import Scene.Shading;
__exported import Utils.Timing.GpuTimer;
__exported import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.EmissivePowerSampler;
import Rendering.Lights.EnvMapSampler;

import Scene.SceneTypes;
import Scene.ShadingData;
import Rendering.Materials.IMaterialInstance;
import Utils.Math.MathHelpers;
import Utils.Math.FormatConversion;
__exported import Scene.Material.ShadingUtils;

import Utils.Color.ColorHelpers;

import RenderPasses.ReSTIRPass.Core.Reservoir;
import RenderPasses.ReSTIRPass.Common.SurfaceData;
import RenderPasses.ReSTIRPass.Common.Utilities;
import RenderPasses.ReSTIRPass.DirectIllumination.LightSampling;

struct LoadSurfaceDataPass
{

    uint2 gFrameDim;  ///< Frame dimensions.
    uint gFrameCount; ///< Frame count since scene was loaded.

    // Resources
    Texture2D<PackedHitInfo> gVBuffer; ///< Packed visibility buffer
    RWStructuredBuffer<PackedSurfaceData> gSurfaceData; ///< Packed surface data
    RWStructuredBuffer<PackedNormalDepth> gNormalDepth; ///< Packed normal depth

    // Debug
	RWTexture2D<float4> gDebug;

	void execute(const uint2 pixel)
    {

	    if (any(pixel >= gFrameDim)) return;
        ShadingData sd;
        let lod = ExplicitLodTextureSampler(0.f);
        uint bufferIndex = getBufferIndex(pixel, gFrameDim);

        // Check if pixel represents a valid primary hit.
		    if(loadShadingData(pixel, gFrameDim, gScene.camera, gVBuffer, sd))
        {
			const float3 primaryRayOrigin = gScene.camera.getPosition();
			const float3 primaryRayDir = getPrimaryRayDir(pixel, gFrameDim, gScene.camera);

            const float3 viewVec = -normalize(primaryRayDir);

			// Create material instance at shading point.
			let mi = gScene.materials.getMaterialInstance(sd, lod);

            // Create surface data from the shading data and material instance.
            SurfaceData surfaceData = SurfaceData::create(sd, mi, primaryRayOrigin);
            gSurfaceData[bufferIndex] = surfaceData.pack();

            // Pack normal and depth
            NormalDepth normalDepth = NormalDepth::create(sd.frame.N, distance(sd.posW, primaryRayOrigin));
            gNormalDepth[bufferIndex] = normalDepth.pack();
        } else {
            // Create invalid surface data.
            SurfaceData invalidSurfaceData = SurfaceData::createInvalid();
            gSurfaceData[bufferIndex] = invalidSurfaceData.pack();
            NormalDepth invalidNormalDepth = NormalDepth::createInvalid();
            gNormalDepth[bufferIndex] = invalidNormalDepth.pack();
        }
	}
}

cbuffer CB
{
    LoadSurfaceDataPass gLoadSurfaceDataPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{

    gLoadSurfaceDataPass.execute(dispatchThreadId.xy);
}
