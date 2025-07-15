add_plugin(ReSTIRPass)

target_sources(ReSTIRPass PRIVATE
    # Main pass files
    ReSTIRPass.cpp
    ReSTIRPass.h

    # Core algorithm files
    Core/Reservoir.slang
    Core/ReservoirGI.slang
    Core/TemporalReuse.slang
    Core/SpatialReuse.slang

    # Direct illumination files
    DirectIllumination/InitialSampling.slang
    DirectIllumination/LightSampling.slang
    DirectIllumination/LightTiling.slang
    DirectIllumination/DirectShading.slang
    DirectIllumination/DirectLightSamples.slang

    # Common utilities
    Common/SurfaceData.slang
    Common/Utilities.slang
    Common/SurfaceLoading.slang
    Common/RayTracing.slang
)

target_copy_shaders(ReSTIRPass RenderPasses/ReSTIRPass)

target_source_group(ReSTIRPass "RenderPasses")
