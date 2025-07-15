/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "ReSTIRPass.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"
#include "Utils/Logger.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/Color/ColorHelpers.slang"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIRPass>();
}

namespace
{
    const std::string kCreateLightTilesPassFilename = "RenderPasses/ReSTIRPass/DirectIllumination/LightTiling.slang";
    const std::string kLoadSurfaceDataPassFilename = "RenderPasses/ReSTIRPass/Common/SurfaceLoading.slang";
    const std::string kGenerateInitialCandidatesPassFilename = "RenderPasses/ReSTIRPass/DirectIllumination/InitialSampling.slang";
    const std::string kTemporalReusePassFilename = "RenderPasses/ReSTIRPass/Core/TemporalReuse.slang";
    const std::string kSpatialReusePassFilename = "RenderPasses/ReSTIRPass/Core/SpatialReuse.slang";
    const std::string kCreateDirectLightSampleFilename = "RenderPasses/ReSTIRPass/DirectIllumination/DirectLightSamples.slang";
    const std::string kShadePassFilename = "RenderPasses/ReSTIRPass/DirectIllumination/DirectShading.slang";


    const std::string kTracePassFilename = "RenderPasses/ReSTIRPass/Common/RayTracing.slang";

    const std::string kShaderModel = "6_5";

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    const uint32_t kMaxPayloadSizeBytes = 100u;// 72u;
    const uint32_t kMaxRecursionDepth = 2u;

    const std::string kInputVBuffer = "vbuffer";
    const std::string kInputMotionVectors = "mvec";

    const ChannelList kInputChannels =
    {
        { kInputVBuffer,    "gVBuffer",     "Visibility buffer in packed format" },
        { kInputMotionVectors,  "gMotionVectors",   "Motion vector buffer (float format)", false /* optional */ },
    };

    const std::string kOutputColor = "color";
    const std::string kOutputAlbedo = "albedo";
    const std::string kDebug = "debug";

    const ChannelList kOutputChannels =
    {
        { kOutputColor, "",     "Output color", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputAlbedo, "",     "Output albedo", true /* optional */, ResourceFormat::RGBA32Float },

        { kDebug, "", "", true /* optional */, ResourceFormat::RGBA32Float },
    };

    const char kMaxBounces[] = "maxBounces";
    const char kComputeDirect[] = "computeDirect";
    const char kUseImportanceSampling[] = "useImportanceSampling";
    const std::string kEmissiveSampler = "emissiveSampler";


    // ReSTIR Options

    Gui::DropdownList kModeList =
    {
        { (uint32_t)ReSTIRPass::Mode::NoResampling, "No resampling" },
        { (uint32_t)ReSTIRPass::Mode::SpatialResampling, "Spatial resampling only" },
        { (uint32_t)ReSTIRPass::Mode::TemporalResampling, "Temporal resampling only" },
        { (uint32_t)ReSTIRPass::Mode::SpatiotemporalResampling, "Spatiotemporal resampling" },

    };

    Gui::DropdownList kBiasCorrectionList =
    {
        { (uint32_t)ReSTIRPass::BiasCorrection::Off, "Off" },
        { (uint32_t)ReSTIRPass::BiasCorrection::Naive, "Naive" },
        { (uint32_t)ReSTIRPass::BiasCorrection::MIS, "MIS" },
        { (uint32_t)ReSTIRPass::BiasCorrection::RayTraced, "RayTraced" },
    };

    Gui::DropdownList kLightTileScreenSize =
    {
        { 1, "1" },
        { 2, "2" },
        { 4, "4" },
        { 8, "8" },
        { 16, "16" },
        { 32, "32" },
        { 64, "64" },
        { 128, "128" },
    };

    const uint32_t kMinLightCandidateCount = 0;
    const uint32_t kMaxLightCandidateCount = 256;

    const uint32_t kMinSpatialIterationCount = 1;
    const uint32_t kMaxSpatialIterationCount = 5;

    const uint32_t kMinSpatialReuseSampleCount = 1;
    const uint32_t kMaxSpatialReuseSampleCount = 20;

    const uint32_t kMinTemporalHistoryLength = 1;
    const uint32_t kMaxTemporalHistoryLength = 40;

    const float kMinSpatialReuseSampleRadius = 0.f;
    const float kMaxSpatialReuseSampleRadius = 60.f;

    const uint32_t kMinLightTileCount = 1;
    const uint32_t kMaxLightTileCount = 1024;


    const uint32_t kMinLightTileSize = 128;
    const uint32_t kMaxLightTileSize = 8096;



}

ReSTIRPass::ReSTIRPass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
    {
        throw RuntimeError("ReSTIRPass: Shader Model 6.5 is not supported by the current device");
    }
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
    {
        throw RuntimeError("ReSTIRPass: Raytracing Tier 1.1 is not supported by the current device");
    }

    parseProperties(props);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);
    FALCOR_ASSERT(mpSampleGenerator);
}

void ReSTIRPass::parseProperties(const Properties& props)
{
}

Properties ReSTIRPass::getProperties() const
{
    return Properties();
}

RenderPassReflection ReSTIRPass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void ReSTIRPass::setFrameDim(const uint2 frameDim)
{
    auto prevFrameDim = frameDim;

    mFrameDim = frameDim;

    if (any(mFrameDim != prevFrameDim))
    {
        mVarsChanged = true;
    }
}

void ReSTIRPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Set new scene.
    mpScene = pScene;

    mFrameCount = 0;
    mFrameDim = {};

    // Need to recreate the trace pass because the shader binding table changes.
    mpTracePass = nullptr;

    resetLighting();

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("ReSTIRPass: This render pass does not support custom primitives.");
        }
        mRecompile = true;
    }
}

void ReSTIRPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!beginFrame(pRenderContext, renderData)) return;

    // Update shader program specialization.
    updatePrograms();

    // Prepare resources.
    prepareResources(pRenderContext, renderData);

    // This should be called after all resources have been created.
    prepareRenderPass(renderData);

    loadSurfaceDataPass(pRenderContext, renderData);

    switch (mReSTIRParams.mode)
    {
    case Mode::NoResampling:
        createLightTiles(pRenderContext);
        generateInitialCandidatesPass(pRenderContext, renderData);
        createDirectSamplesPass(pRenderContext, renderData);
        shadePass(pRenderContext, renderData);
        break;
    case Mode::SpatialResampling:
        createLightTiles(pRenderContext);
        generateInitialCandidatesPass(pRenderContext, renderData);
        spatialReusePass(pRenderContext, renderData);
        createDirectSamplesPass(pRenderContext, renderData);
        shadePass(pRenderContext, renderData);
        break;
    case Mode::TemporalResampling:
        createLightTiles(pRenderContext);
        generateInitialCandidatesPass(pRenderContext, renderData);
        temporalReusePass(pRenderContext, renderData);
        createDirectSamplesPass(pRenderContext, renderData);
        shadePass(pRenderContext, renderData);
        break;
    case Mode::SpatiotemporalResampling:
        createLightTiles(pRenderContext);
        generateInitialCandidatesPass(pRenderContext, renderData);
        temporalReusePass(pRenderContext, renderData);
        spatialReusePass(pRenderContext, renderData);
        createDirectSamplesPass(pRenderContext, renderData);
        shadePass(pRenderContext, renderData);
        break;


    }

    endFrame(pRenderContext, renderData);
}

void ReSTIRPass::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    // Rendering options.
    dirty |= renderRenderingUI(widget);

    if (dirty)
    {
        mOptionsChanged = true;
    }
}

bool ReSTIRPass::renderRenderingUI(Gui::Widgets& widget)
{
    bool dirty = false;

    bool temporalResampling = (mReSTIRParams.mode == Mode::TemporalResampling || mReSTIRParams.mode == Mode::SpatiotemporalResampling);
    bool spatialResampling = (mReSTIRParams.mode == Mode::SpatialResampling || mReSTIRParams.mode == Mode::SpatiotemporalResampling);

    dirty |= widget.dropdown("Mode", kModeList, reinterpret_cast<uint32_t&>(mReSTIRParams.mode));

    if (auto group = widget.group("Precomputed light tiles", false))
    {

        dirty |= group.var("Light tile count", mReSTIRParams.lightTileCount, kMinLightTileCount, kMaxLightTileCount);
        group.tooltip("The number of light tiles created in the presampling phase.");

        dirty |= group.var("Light tile size", mReSTIRParams.lightTileSize, kMinLightTileSize, kMaxLightTileSize);
        group.tooltip("The size of single light tile created in the presampling phase.");

        dirty |= group.dropdown("Light tile screen size", kLightTileScreenSize, reinterpret_cast<uint32_t&>(mReSTIRParams.lightTileScreenSize));
        group.tooltip("The size of screen tile in pixels which form a group accessing the same light tile.");
    }

    if (auto group = widget.group("Initial resampling", false))
    {
        dirty |= group.var("Emissive light samples", mReSTIRParams.emissiveLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount);
        group.tooltip("Number of initial emissive light candidate samples.");

        dirty |= group.var("Environment light samples", mReSTIRParams.envLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount);
        group.tooltip("Number of initial environment light candidate samples.");

        dirty |= group.var("Analytic light samples", mReSTIRParams.analyticLightCandidateCount, kMinLightCandidateCount, kMaxLightCandidateCount);
        group.tooltip("Number of initial analytic light candidate samples.");

        dirty |= group.var("Emissive intensity multiplier", mReSTIRParams.emissiveIntensityMultiplier, 0.1f, 1000.0f);
        group.tooltip("Multiplier for emissive light intensity to make them more visible in the scene.");

        dirty |= group.checkbox("Test initial candidate visibility", mReSTIRParams.testInitialSampleVisibility);
        group.tooltip("Performs a visibility test for the selected initial candidate.");

    }

    if (temporalResampling)
    {
        if (auto group = widget.group("Temporal resampling", false))
        {
            dirty |= group.var("Max history length", mReSTIRParams.temporalHistoryLength, kMinTemporalHistoryLength, kMaxTemporalHistoryLength);
            group.tooltip("Maximum history length for temporal reuse [frames]. This should be lower (<4) for the ray traced unbiased reuse to ensure correct convergence.");
        }
    }

    if (spatialResampling)
    {
        if (auto group = widget.group("Spatial resampling", false))
        {

            dirty |= group.var("Iterations", mReSTIRParams.spatialIterationCount, kMinSpatialIterationCount, kMaxSpatialIterationCount);
            group.tooltip("Number of spatial reuse iterations.");

            dirty |= group.var("Sample count", mReSTIRParams.spatialReuseSampleCount, kMinSpatialReuseSampleCount, kMaxSpatialReuseSampleCount);
            group.tooltip("Number of neighbor samples considered for resampling.");

            dirty |= group.var("Sample radius", mReSTIRParams.spatialReuseSampleRadius, kMinSpatialReuseSampleRadius, kMaxSpatialReuseSampleRadius);
            group.tooltip("Screen-space radius for neighbor selection in pixels.");

            dirty |= group.var("Visibility test threshold", mReSTIRParams.spatialVisibilityThreshold, 0.f, mReSTIRParams.spatialReuseSampleRadius);
            group.tooltip("Distance from the pixel after which the visibility test is performed.");
        }
    }

    if (spatialResampling || temporalResampling)
    {
        if (auto group = widget.group("Resampling options", false))
        {
            dirty |= group.dropdown("Bias correction", kBiasCorrectionList, reinterpret_cast<uint32_t&>(mReSTIRParams.biasCorrection));
            group.tooltip("Type of correction to prevent the occurrence of bias.");

            dirty |= group.var("Depth threshold", mReSTIRParams.depthThreshold, 0.f, 1.f);
            group.tooltip("Depth threshold for sample reuse.");

            dirty |= group.var("Normal threshold", mReSTIRParams.normalThreshold, 0.f, 1.f);
            group.tooltip("Normal threshold for sample reuse.");
        }
    }



    if (dirty) mRecompile = true;
    return dirty;
}

void ReSTIRPass::prepareRenderPass(const RenderData& renderData)
{
    // Bind resources.
    auto var = mpTracePass->pVars->getRootVar();
    setShaderData(var, renderData);
}

void ReSTIRPass::setShaderData(const ShaderVar& var, const RenderData& renderData, bool useLightSampling) const
{
    var["CB"]["gFrameCount"] = mFrameCount;

    var["gVBuffer"] = renderData.getTexture(kInputVBuffer);

    if (useLightSampling && mpEmissiveSampler)
    {
        // TODO: Do we have to bind this every frame?
        mpEmissiveSampler->bindShaderData(var["CB"]["gEmissiveLightSampler"]);
    }
}

void ReSTIRPass::tracePass(RenderContext* pRenderContext, const RenderData& renderData, TracePass& tracePass)
{
    FALCOR_PROFILE(pRenderContext, tracePass.name);

    FALCOR_ASSERT(tracePass.pProgram != nullptr && tracePass.pBindingTable != nullptr && tracePass.pVars != nullptr);

    // Bind global resources.
    auto var = tracePass.pVars->getRootVar();
    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"]);

    if (mVarsChanged) mpSampleGenerator->bindShaderData(var);
    var["gGIReservoirs"] = mpGIReservoirs;
    var["gDebug"] = renderData.getTexture(kDebug);
    // Full screen dispatch.
    mpScene->raytrace(pRenderContext, tracePass.pProgram.get(), tracePass.pVars, { mFrameDim, 1u });
}

void ReSTIRPass::createLightTiles(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "createLightTilesPass");

    auto var = mpCreateLightTiles->getRootVar()["CB"]["gCreateLightTilesPass"];

    var["gLightTiles"] = mpLightTiles;
    var["gFrameCount"] = mFrameCount;

    if (mpEmissiveGeometryAliasTable) mpEmissiveGeometryAliasTable->bindShaderData(var["gLightSampler"]["emissiveGeometryAliasTable"]);
    if (mpAnalyticLightsAliasTable) mpAnalyticLightsAliasTable->bindShaderData(var["gLightSampler"]["analyticLightsAliasTable"]);
    if (mpEnvironmentAliasTable)
    {
        mpEnvironmentAliasTable->bindShaderData(var["gLightSampler"]["environmentAliasTable"]);
        var["gLightSampler"]["environmentLuminanceTable"] = mpEnvironmentLuminanceTable;
    }

    mpScene->bindShaderData(mpCreateLightTiles->getRootVar()["gScene"]);

    mpCreateLightTiles->execute(pRenderContext, uint3(mReSTIRParams.lightTileSize, mReSTIRParams.lightTileCount, 1));
}

void ReSTIRPass::loadSurfaceDataPass(RenderContext* pRenderContext, const RenderData& renderData)
{

    FALCOR_PROFILE(pRenderContext, "loadSurfaceDataPass");

    // Bind resources.
    auto var = mpLoadSurfaceDataPass->getRootVar()["CB"]["gLoadSurfaceDataPass"];

    var["gFrameDim"] = mFrameDim;
    var["gFrameCount"] = mFrameCount;

    var["gVBuffer"] = renderData.getTexture(kInputVBuffer);
    var["gSurfaceData"] = mpSurfaceData;
    var["gNormalDepth"] = mpNormalDepth;

    var["gDebug"] = renderData.getTexture(kDebug);


    mpScene->bindShaderData(mpLoadSurfaceDataPass->getRootVar()["gScene"]);
    mpLoadSurfaceDataPass->execute(pRenderContext, { mFrameDim.x, mFrameDim.y, 1u });
}

void ReSTIRPass::generateInitialCandidatesPass(RenderContext* pRenderContext, const RenderData& renderData)
{

    FALCOR_PROFILE(pRenderContext, "generateInitialCandidatesPass");

    // Bind resources.
    auto var = mpGenerateInitialCandidatesPass->getRootVar()["CB"]["gGenerateInitialCandidatesPass"];

    // Bind static resources that don't change per frame.
    if (mVarsChanged)
    {
        if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(var["gEnvMapSampler"]);
    }

    var["gFrameDim"] = mFrameDim;
    var["gFrameCount"] = mFrameCount;

    var["gSurfaceData"] = mpSurfaceData;
    var["gReservoirs"] = mpReservoirs;

    var["gLightTiles"] = mpLightTiles;

    var["gDebug"] = renderData.getTexture(kDebug);

    if (mpEmissiveGeometryAliasTable)
    {
        mpEmissiveGeometryAliasTable->bindShaderData(var["gLightSampler"]["emissiveGeometryAliasTable"]);
    }
    if (mpAnalyticLightsAliasTable)
    {
        mpAnalyticLightsAliasTable->bindShaderData(var["gLightSampler"]["analyticLightsAliasTable"]);
    }
    if (mpEnvironmentAliasTable)
    {
        mpEnvironmentAliasTable->bindShaderData(var["gLightSampler"]["environmentAliasTable"]);
        var["gLightSampler"]["environmentLuminanceTable"] = mpEnvironmentLuminanceTable;
    }

    mpScene->bindShaderData(mpGenerateInitialCandidatesPass->getRootVar()["gScene"]);
    mpGenerateInitialCandidatesPass->execute(pRenderContext, { !mReSTIRParams.useCheckerboarding ? mFrameDim.x : mFrameDim.x / 2, mFrameDim.y, 1u });
}

void ReSTIRPass::temporalReusePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "temporalReusePass");

    // Bind resources.
    auto var = mpTemporalReusePass->getRootVar()["CB"]["gTemporalReusePass"];

    var["gFrameDim"] = mFrameDim;
    var["gFrameCount"] = mFrameCount;

    var["gMotionVectors"] = renderData.getTexture(kInputMotionVectors);
    var["gReservoirs"] = mpReservoirs;
    var["gSurfaceData"] = mpSurfaceData;
    var["gNormalDepth"] = mpNormalDepth;

    var["gPrevSurfaceData"] = mpPrevSurfaceData;
    var["gPrevNormalDepth"] = mpPrevNormalDepth;
    var["gPrevReservoirs"] = mpPrevReservoirs;
    var["gDebug"] = renderData.getTexture(kDebug);

    if (mpEmissiveGeometryAliasTable)
    {
        mpEmissiveGeometryAliasTable->bindShaderData(var["gLightSampler"]["emissiveGeometryAliasTable"]);
    }
    if (mpAnalyticLightsAliasTable)
    {
        mpAnalyticLightsAliasTable->bindShaderData(var["gLightSampler"]["analyticLightsAliasTable"]);
    }
    if (mpEnvironmentAliasTable)
    {
        mpEnvironmentAliasTable->bindShaderData(var["gLightSampler"]["environmentAliasTable"]);
        var["gLightSampler"]["environmentLuminanceTable"] = mpEnvironmentLuminanceTable;
    }

    mpScene->bindShaderData(mpTemporalReusePass->getRootVar()["gScene"]);

    mpTemporalReusePass->execute(pRenderContext, { mFrameDim, 1u });
}

void ReSTIRPass::spatialReusePass(RenderContext* pRenderContext, const RenderData& renderData)
{

    FALCOR_PROFILE(pRenderContext, "spatialReusePass");

    for (size_t iteration = 0; iteration < mReSTIRParams.spatialIterationCount; iteration++)
    {
        // Bind resources.
        auto var = mpSpatialReusePass->getRootVar()["CB"]["gSpatialReusePass"];

        var["gFrameDim"] = mFrameDim;
        var["gFrameCount"] = mFrameCount;

        var["gSurfaceData"] = mpSurfaceData;
        var["gNormalDepth"] = mpNormalDepth;

        std::swap(mpReservoirs, mpPrevReservoirs);

        var["gReservoirs"] = mpPrevReservoirs;
        var["gOutReservoirs"] = mpReservoirs;
        var["gDebug"] = renderData.getTexture(kDebug);

        if (mpEmissiveGeometryAliasTable)
        {
            mpEmissiveGeometryAliasTable->bindShaderData(var["gLightSampler"]["emissiveGeometryAliasTable"]);
        }
        if (mpAnalyticLightsAliasTable)
        {
            mpAnalyticLightsAliasTable->bindShaderData(var["gLightSampler"]["analyticLightsAliasTable"]);
        }
        if (mpEnvironmentAliasTable)
        {
            mpEnvironmentAliasTable->bindShaderData(var["gLightSampler"]["environmentAliasTable"]);
            var["gLightSampler"]["environmentLuminanceTable"] = mpEnvironmentLuminanceTable;
        }

        mpScene->bindShaderData(mpSpatialReusePass->getRootVar()["gScene"]);

        mpSpatialReusePass->execute(pRenderContext, { mFrameDim, 1u });
    }

}

void ReSTIRPass::createDirectSamplesPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "createDirectLightSamplesPass");

    // Bind resources.
    auto var = mpCreateDirectLightSamplesPass->getRootVar()["CB"]["gCreateDirectLightSamplesPass"];

    var["gFrameDim"] = mFrameDim;
    var["gFrameCount"] = mFrameCount;

    var["gNormalDepth"] = mpNormalDepth;
    var["gSurfaceData"] = mpSurfaceData;
    var["gReservoirs"] = mpReservoirs;
    var["gDirectLightSamples"] = mpDirectLightSamples;

    var["gDebug"] = renderData.getTexture(kDebug);

    if (mpEmissiveGeometryAliasTable)
    {
        mpEmissiveGeometryAliasTable->bindShaderData(var["gLightSampler"]["emissiveGeometryAliasTable"]);
    }
    if (mpAnalyticLightsAliasTable)
    {
        mpAnalyticLightsAliasTable->bindShaderData(var["gLightSampler"]["analyticLightsAliasTable"]);
    }
    if (mpEnvironmentAliasTable)
    {
        mpEnvironmentAliasTable->bindShaderData(var["gLightSampler"]["environmentAliasTable"]);
        var["gLightSampler"]["environmentLuminanceTable"] = mpEnvironmentLuminanceTable;
    }

    mpScene->bindShaderData(mpCreateDirectLightSamplesPass->getRootVar()["gScene"]);

    mpCreateDirectLightSamplesPass->execute(pRenderContext, { mFrameDim, 1u });
}

void ReSTIRPass::shadePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "shadePass");

    // Bind resources.
    auto var = mpShadePass->getRootVar()["CB"]["gShadePass"];

    var["gFrameDim"] = mFrameDim;
    var["gFrameCount"] = mFrameCount;

    var["gVBuffer"] = renderData.getTexture(kInputVBuffer);
    var["gDirectLightSamples"] = mpDirectLightSamples;

    var["gOutputColor"] = renderData.getTexture(kOutputColor);
    var["gOutputAlbedo"] = renderData.getTexture(kOutputAlbedo);

    var["gDebug"] = renderData.getTexture(kDebug);


    mpScene->bindShaderData(mpShadePass->getRootVar()["gScene"]);

    mpShadePass->execute(pRenderContext, { mFrameDim, 1u });
}


ReSTIRPass::TracePass::TracePass(ref<Device> pDevice, const std::string& name, const std::string& passDefine, const ref<Scene>& pScene, const DefineList& defines, const TypeConformanceList& globalTypeConformances)
    : name(name)
    , passDefine(passDefine)
{

    ProgramDesc desc;
    desc.addShaderModules(pScene->getShaderModules());
    desc.addShaderLibrary(kTracePassFilename);
    desc.setShaderModel(ShaderModel::SM6_5);
    desc.setMaxPayloadSize(kMaxPayloadSizeBytes); // TODO: The required minimum is 72 bytes!
    desc.setMaxAttributeSize(pScene->getRaytracingMaxAttributeSize());
    desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
    if (!pScene->hasProceduralGeometry()) desc.setRtPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

    // Create ray tracing binding table.
    pBindingTable = RtBindingTable::create(2, 2, pScene->getGeometryCount());

    // Specify entry point for raygen and miss shaders.
    // The raygen shader needs type conformances for *all* materials in the scene.
    // The miss shader doesn't need type conformances as it doesn't access any materials.
    pBindingTable->setRayGen(desc.addRayGen("rayGen", globalTypeConformances));
    pBindingTable->setMiss(0, desc.addMiss("scatterMiss"));
    pBindingTable->setMiss(1, desc.addMiss("shadowMiss"));

    auto materialTypes = pScene->getMaterialSystem().getMaterialTypes();

    for (const auto& materialType : materialTypes)
    {
        auto typeConformances = pScene->getMaterialSystem().getTypeConformances(materialType);

        // Add hit groups for triangles.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::TriangleMesh, materialType); !geometryIDs.empty())
        {
            pBindingTable->setHitGroup(0, geometryIDs, desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit", "", typeConformances, to_string(materialType)));
            pBindingTable->setHitGroup(1, geometryIDs, desc.addHitGroup("", "shadowTriangleMeshAnyHit", "", typeConformances, to_string(materialType)));
        }
    }

    pProgram = Program::create(pDevice, desc, defines);

}

void ReSTIRPass::TracePass::prepareProgram(ref<Device> pDevice, const DefineList& defines)
{
    FALCOR_ASSERT(pProgram != nullptr && pBindingTable != nullptr);
    pProgram->addDefines(defines);
    if (!passDefine.empty()) pProgram->addDefine(passDefine);
    pVars = RtProgramVars::create(pDevice, pProgram, pBindingTable);
}

void ReSTIRPass::updatePrograms()
{
    FALCOR_ASSERT(mpScene);

    if (mRecompile == false) return;

    auto defines = mStaticParams.getDefines(*this);
    auto globalTypeConformances = mpScene->getMaterialSystem().getTypeConformances();

    // Create trace passes lazily.
    if (!mpTracePass) mpTracePass = std::make_unique<TracePass>(mpDevice, "tracePass", "", mpScene, defines, globalTypeConformances);

    // Create program vars for trace programs.
    // We only need to set defines for program specialization here. Type conformances have already been setup on construction.
    mpTracePass->prepareProgram(mpDevice, defines);

    ProgramDesc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(globalTypeConformances);
    baseDesc.setShaderModel(ShaderModel::SM6_5);

    if (!mpCreateLightTiles)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kCreateLightTilesPassFilename).csEntry("main");
        mpCreateLightTiles = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpLoadSurfaceDataPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kLoadSurfaceDataPassFilename).csEntry("main");
        mpLoadSurfaceDataPass = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpGenerateInitialCandidatesPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kGenerateInitialCandidatesPassFilename).csEntry("main");
        mpGenerateInitialCandidatesPass = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpTemporalReusePass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kTemporalReusePassFilename).csEntry("main");
        mpTemporalReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpSpatialReusePass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kSpatialReusePassFilename).csEntry("main");
        mpSpatialReusePass = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpCreateDirectLightSamplesPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kCreateDirectLightSampleFilename).csEntry("main");
        mpCreateDirectLightSamplesPass = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpShadePass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kShadePassFilename).csEntry("main");
        mpShadePass = ComputePass::create(mpDevice, desc, defines, false);
    }

    // Perform program specialization.
    // Note that we must use set instead of add functions to replace any stale state.
    auto prepareProgram = [&](ref<Program> program)
    {
        program->addDefines(defines);
    };

    prepareProgram(mpCreateLightTiles->getProgram());
    prepareProgram(mpLoadSurfaceDataPass->getProgram());
    prepareProgram(mpGenerateInitialCandidatesPass->getProgram());
    prepareProgram(mpTemporalReusePass->getProgram());
    prepareProgram(mpSpatialReusePass->getProgram());
    prepareProgram(mpCreateDirectLightSamplesPass->getProgram());
    prepareProgram(mpShadePass->getProgram());

    mpCreateLightTiles->setVars(nullptr);
    mpLoadSurfaceDataPass->setVars(nullptr);
    mpGenerateInitialCandidatesPass->setVars(nullptr);
    mpTemporalReusePass->setVars(nullptr);
    mpSpatialReusePass->setVars(nullptr);
    mpCreateDirectLightSamplesPass->setVars(nullptr);
    mpShadePass->setVars(nullptr);

    mVarsChanged = true;
    mRecompile = false;
}

void ReSTIRPass::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint32_t pixelCount = mFrameDim.x * mFrameDim.y;

    // Create reservoir buffers.
    if (!mpReservoirs || mpReservoirs->getElementCount() < pixelCount)
    {
        mpReservoirs = mpDevice->createStructuredBuffer(sizeof(uint4), pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }
    if (!mpPrevReservoirs || mpPrevReservoirs->getElementCount() < pixelCount)
    {
        mpPrevReservoirs = mpDevice->createStructuredBuffer(sizeof(uint4), pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }

    // Create surface data buffers.
    if (!mpSurfaceData || mpSurfaceData->getElementCount() < pixelCount)
    {
        mpSurfaceData = mpDevice->createStructuredBuffer(sizeof(uint4) * 2, pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }
    if (!mpPrevSurfaceData || mpPrevSurfaceData->getElementCount() < pixelCount)
    {
        mpPrevSurfaceData = mpDevice->createStructuredBuffer(sizeof(uint4) * 2, pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }

    // Create normal depth buffers.
    if (!mpNormalDepth || mpNormalDepth->getElementCount() < pixelCount)
    {
        mpNormalDepth = mpDevice->createStructuredBuffer(sizeof(uint2), pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }
    if (!mpPrevNormalDepth || mpPrevNormalDepth->getElementCount() < pixelCount)
    {
        mpPrevNormalDepth = mpDevice->createStructuredBuffer(sizeof(uint2), pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }

    // Create light tile buffers.
    uint32_t elementCount = mReSTIRParams.lightTileCount * mReSTIRParams.lightTileSize;
    if (!mpLightTiles || mpLightTiles->getElementCount() < elementCount)
    {
        mpLightTiles = mpDevice->createStructuredBuffer(sizeof(uint4) * 2, elementCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }

    if (!mpDirectLightSamples || mpDirectLightSamples->getElementCount() < pixelCount)
    {
        mpDirectLightSamples = mpDevice->createStructuredBuffer(sizeof(uint4), pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }

    // Create GI reservoirs
    if (mReSTIRParams.mode == Mode::ReSTIRGI && (!mpGIReservoirs || mpGIReservoirs->getElementCount() < pixelCount))
    {
        mpGIReservoirs = mpDevice->createStructuredBuffer(sizeof(uint4) * 4, pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }
    if (mReSTIRParams.mode == Mode::ReSTIRGI && (!mpPrevGIReservoirs || mpPrevGIReservoirs->getElementCount() < pixelCount))
    {
        mpPrevGIReservoirs = mpDevice->createStructuredBuffer(sizeof(uint4) * 4, pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }
    if (mReSTIRParams.mode == Mode::ReSTIRGI && (!mpSpatialGIReservoirs || mpSpatialGIReservoirs->getElementCount() < pixelCount))
    {
        mpSpatialGIReservoirs = mpDevice->createStructuredBuffer(sizeof(uint4) * 4, pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false);
    }
}

bool ReSTIRPass::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    mpScene->getLightCollection(pRenderContext)->prepareSyncCPUData(pRenderContext); // TODO: is this necassary?
    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RenderSettingsChanged))
    {
        lightingChanged = true;
        mRecompile = true;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::SDFGridConfigChanged))
    {
        mRecompile = true;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::EnvMapChanged))
    {
        mpEnvironmentAliasTable = nullptr;
        lightingChanged = true;
        mRecompile = true;
    }

    if (mpScene->useEnvLight())
    {
        if (!mpEnvironmentAliasTable || !mpEnvironmentLuminanceTable)
        {
            const auto& environmentMapTex = mpScene->getEnvMap()->getEnvMap();
            mpEnvironmentAliasTable = createEnvironmentAliasTable(pRenderContext, environmentMapTex);
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEnvironmentAliasTable)
        {
            mpEnvironmentAliasTable = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->useEmissiveLights())
    {

        if (!mpEmissiveGeometryAliasTable)
        {
            auto lightCollection = mpScene->getLightCollection(pRenderContext);
            lightCollection->update(pRenderContext);
            if (lightCollection->getActiveLightCount(pRenderContext) > 0)
            {
                mpEmissiveGeometryAliasTable = createEmissiveGeometryAliasTable(pRenderContext, lightCollection);
                lightingChanged = true;
                mRecompile = true;
            }
        }

        if (mReSTIRParams.mode == Mode::ReSTIRGI)
        {
            if (!mpEmissiveSampler)
            {
                const auto& pLights = mpScene->getLightCollection(pRenderContext);
                FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
                FALCOR_ASSERT(!mpEmissiveSampler);

                mpEmissiveSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
                lightingChanged = true;
                mRecompile = true;
            }
        }
        else
        {
            if (mpEmissiveSampler)
            {
                mpEmissiveSampler = nullptr;
                lightingChanged = true;
                mRecompile = true;
            }
        }
    }
    else
    {
        if (mpEmissiveGeometryAliasTable)
        {
            mpEmissiveSampler = nullptr;
            mpEmissiveGeometryAliasTable = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    if (mpScene->useAnalyticLights())
    {
        if (!mpAnalyticLightsAliasTable)
        {
            if (mpScene->getLightCount() > 0)
            {
                mpAnalyticLightsAliasTable = createAnalyticLightsAliasTable(pRenderContext);
                lightingChanged = true;
                mRecompile = true;
            }
        }
    }
    else
    {
        if (mpAnalyticLightsAliasTable)
        {
            mpAnalyticLightsAliasTable = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
        auto defines = mpEmissiveSampler->getDefines();
        if (mpTracePass && mpTracePass->pProgram->addDefines(defines)) mRecompile = true;
    }

    return lightingChanged;
}

std::unique_ptr<AliasTable> ReSTIRPass::createEmissiveGeometryAliasTable(RenderContext* pRenderContext, const ref<LightCollection>& lightCollection)
{
    assert(lightCollection);

    lightCollection->update(pRenderContext);

    const auto& triangles = lightCollection->getMeshLightTriangles(pRenderContext);

    std::vector<float> weights(triangles.size());

    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] = luminance(triangles[i].averageRadiance) * triangles[i].area;
    }

    return std::make_unique<AliasTable>(mpDevice, std::move(weights), mRnd);
}

std::unique_ptr<AliasTable> ReSTIRPass::createEnvironmentAliasTable(RenderContext* pRenderContext, const ref<Texture>& envTexture)
{
    assert(envTexture);

    uint32_t width = envTexture->getWidth();
    uint32_t height = envTexture->getHeight();
    uint32_t texelCount = width * height;
    uint32_t channelCount = getFormatChannelCount(envTexture->getFormat());

    if (getFormatType(envTexture->getFormat()) != FormatType::Float) return nullptr;

    std::vector<uint8_t> rawTexture = pRenderContext->readTextureSubresource(envTexture.get(), 0);
    const float* texels = reinterpret_cast<const float*>(rawTexture.data());

    std::vector<float> envMapLuminances(texelCount);

    if (channelCount == 1)
    {
        for (size_t i = 0; i < texelCount; i++)
        {
            envMapLuminances[i] = texels[i * channelCount];
        }
    }
    else if (channelCount == 3 || channelCount == 4)
    {
        for (size_t i = 0; i < texelCount; i++)
        {
            envMapLuminances[i] = luminance(float3(texels[i * channelCount], texels[i * channelCount + 1], texels[i * channelCount + 2]));
        }
    }
    else
    {
        throw std::exception("FAILED TO BUILD ENV ALIAS TABLE: Invalid number of color channels in environment map texture.");
    }

    mpEnvironmentLuminanceTable = mpDevice->createTypedBuffer<float>((uint32_t)envMapLuminances.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, envMapLuminances.data());

    std::vector<float> weights(texelCount);

    for (size_t y = 0; y < height; y++)
    {
        float theta = ((y + 0.5f) / height) * static_cast<float>(M_PI);
        float dPhi = 2.f * static_cast<float>(M_PI) / width;
        float dTheta = static_cast<float>(M_PI) / height;
        float diffSolidAngle = dPhi * dTheta * std::sin(theta);

        for (size_t x = 0; x < width; x++)
        {
            size_t index = y * width + x;
            weights[index] = diffSolidAngle * envMapLuminances[index];
        }
    }

    return std::make_unique<AliasTable>(mpDevice, std::move(weights), mRnd);
}

std::unique_ptr<AliasTable> ReSTIRPass::createAnalyticLightsAliasTable(RenderContext* pRenderContext)
{
    uint32_t lightCount = mpScene->getLightCount();
    std::vector<float> weights(lightCount);
    for (uint32_t i = 0; i < lightCount; i++)
    {
        auto light = mpScene->getLight(i);
        weights[i] = luminance(light->getIntensity());
    }

    return std::make_unique<AliasTable>(mpDevice, std::move(weights), mRnd);
}

void ReSTIRPass::prepareMaterials(RenderContext* pRenderContext)
{
    // This functions checks for material changes and performs any necessary update.
    // For now all we need to do is to trigger a recompile so that the right defines get set.
    // In the future, we might want to do additional material-specific setup here.

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::MaterialsChanged))
    {
        mRecompile = true;
        mClearReservoirs = true;  // Clear reservoir history when materials change to avoid temporal artifacts
    }
}

void ReSTIRPass::resetLighting()
{
    // Retain the options for the emissive sampler.

    mpEnvironmentAliasTable = nullptr;
    mpEnvironmentLuminanceTable = nullptr;
    mpEmissiveGeometryAliasTable = nullptr;
    mpEmissiveSampler = nullptr;
    mpEnvMapSampler = nullptr;
    mRecompile = true;
    mClearReservoirs = true;  // Clear reservoir history when lighting setup changes
}

bool ReSTIRPass::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutputColor = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutputColor);

    // Set output frame dimension
    setFrameDim(uint2(pOutputColor->getWidth(), pOutputColor->getHeight()));

    // Validate all I/O sizes match the expected size.
    // If not, we'll disable the path tracer to give the user a chance to fix the configuration before re-enabling it.
    bool resolutionMismatch = false;
    auto validateChannels = [&](const auto& channels)
    {
        for (const auto& channel : channels)
        {
            auto pTexture = renderData.getTexture(channel.name);
            if (pTexture && (pTexture->getWidth() != mFrameDim.x || pTexture->getHeight() != mFrameDim.y)) resolutionMismatch = true;
        }
    };
    validateChannels(kInputChannels);
    validateChannels(kOutputChannels);

    if (mEnabled && resolutionMismatch)
    {
        logError("Render pass I/O sizes don't match. The pass will be disabled.");
        mEnabled = false;
    }

    if (mpScene == nullptr || !mEnabled)
    {
        pRenderContext->clearUAV(pOutputColor->getUAV().get(), float4(0.f));

        // Set refresh flag if changes that affect the output have occured.
        // This is needed to ensure other passes get notified when the path tracer is enabled/disabled.
        if (mOptionsChanged)
        {
            auto& dict = renderData.getDictionary();
            auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
            if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
            dict[Falcor::kRenderPassRefreshFlags] = flags;
        }

        return false;
    }

    // Update materials.
    prepareMaterials(pRenderContext);

    // Update the env map and emissive sampler to the current frame.
    bool lightingChanged = prepareLighting(pRenderContext);

    // Update refresh flag if changes that affect the output have occured.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged || lightingChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
        if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        if (lightingChanged) flags |= Falcor::RenderPassRefreshFlags::LightingChanged;
        dict[Falcor::kRenderPassRefreshFlags] = flags;
        mOptionsChanged = false;
    }

    return true;
}

void ReSTIRPass::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    mVarsChanged = false;
    mFrameCount++;

    // Swap reservoir data.
    auto copyTexture = [pRenderContext](Texture* pDst, const Texture* pSrc)
    {
        if (pDst && pSrc)
        {
            FALCOR_ASSERT(pDst && pSrc);
            FALCOR_ASSERT(pDst->getFormat() == pSrc->getFormat());
            FALCOR_ASSERT(pDst->getWidth() == pSrc->getWidth() && pDst->getHeight() == pSrc->getHeight());
            pRenderContext->copyResource(pDst, pSrc);
        }
        else if (pDst)
        {
            pRenderContext->clearUAV(pDst->getUAV().get(), uint4(0, 0, 0, 0));
        }
    };

    // Clear reservoir history if materials have changed to avoid temporal artifacts
    if (mClearReservoirs)
    {
        if (mpPrevReservoirs)
            pRenderContext->clearUAV(mpPrevReservoirs->getUAV().get(), uint4(0, 0, 0, 0));
        if (mpPrevSurfaceData)
            pRenderContext->clearUAV(mpPrevSurfaceData->getUAV().get(), uint4(0, 0, 0, 0));
        if (mpPrevNormalDepth)
            pRenderContext->clearUAV(mpPrevNormalDepth->getUAV().get(), uint4(0, 0, 0, 0));
        if (mpPrevGIReservoirs)
            pRenderContext->clearUAV(mpPrevGIReservoirs->getUAV().get(), uint4(0, 0, 0, 0));

        mClearReservoirs = false;
    }
    else
    {
        // Normal reservoir swapping
        std::swap(mpReservoirs, mpPrevReservoirs);
        std::swap(mpSurfaceData, mpPrevSurfaceData);
        std::swap(mpNormalDepth, mpPrevNormalDepth);
        //std::swap(mpGIReservoirs, mpPrevGIReservoirs);
    }
}

DefineList ReSTIRPass::StaticParams::getDefines(const ReSTIRPass& owner) const
{
    DefineList defines;

    // Sampling utilities configuration.
    FALCOR_ASSERT(owner.mpSampleGenerator);
    defines.add(owner.mpSampleGenerator->getDefines());

    if (owner.mpEmissiveSampler) defines.add(owner.mpEmissiveSampler->getDefines());

    // Scene-specific configuration.
    const auto& scene = owner.mpScene;
    if (scene) defines.add(scene->getSceneDefines());
    defines.add("USE_ENV_LIGHT", scene && scene->useEnvLight() ? "1" : "0");
    defines.add("USE_ENV_BACKGROUND", scene && scene->useEnvBackground() ? "1" : "0");
    defines.add("USE_ANALYTIC_LIGHTS", scene && scene->useAnalyticLights() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", scene && scene->useEmissiveLights() ? "1" : "0");

    defines.add("EMISSIVE_LIGHT_CANDIDATE_COUNT", std::to_string(owner.mReSTIRParams.emissiveLightCandidateCount));
    defines.add("ENV_LIGHT_CANDIDATE_COUNT", std::to_string(owner.mReSTIRParams.envLightCandidateCount));
    defines.add("ANALYTIC_LIGHT_CANDIDATE_COUNT", std::to_string(owner.mReSTIRParams.analyticLightCandidateCount));
    defines.add("TEST_INITIAL_SAMPLE_VISIBILITY", std::to_string(owner.mReSTIRParams.testInitialSampleVisibility));

    defines.add("DEPTH_THRESHOLD", std::to_string(owner.mReSTIRParams.depthThreshold));
    defines.add("NORMAL_THRESHOLD", std::to_string(owner.mReSTIRParams.normalThreshold));

    defines.add("EMISSIVE_INTENSITY_MULTIPLIER", std::to_string(owner.mReSTIRParams.emissiveIntensityMultiplier));

    defines.add("TEMPORAL_MAX_HISTORY_LENGTH", std::to_string(owner.mReSTIRParams.temporalHistoryLength));

    defines.add("SPATIAL_REUSE_SAMPLE_COUNT", std::to_string(owner.mReSTIRParams.spatialReuseSampleCount));
    defines.add("SPATIAL_REUSE_SAMPLE_RADIUS", std::to_string(owner.mReSTIRParams.spatialReuseSampleRadius));

    defines.add("UNBIASED_NAIVE", owner.mReSTIRParams.biasCorrection == ReSTIRPass::BiasCorrection::Naive ? "1" : "0");
    defines.add("UNBIASED_MIS", owner.mReSTIRParams.biasCorrection == ReSTIRPass::BiasCorrection::MIS || owner.mReSTIRParams.biasCorrection == ReSTIRPass::BiasCorrection::RayTraced ? "1" : "0");
    defines.add("UNBIASED_RAYTRACED", owner.mReSTIRParams.biasCorrection == ReSTIRPass::BiasCorrection::RayTraced ? "1" : "0");
    defines.add("BIASED", owner.mReSTIRParams.biasCorrection == ReSTIRPass::BiasCorrection::Off ? "1" : "0");

    defines.add("LIGHT_TILE_SIZE", std::to_string(owner.mReSTIRParams.lightTileSize));
    defines.add("LIGHT_TILE_COUNT", std::to_string(owner.mReSTIRParams.lightTileCount));
    defines.add("LIGHT_TILE_SCREEN_SIZE", std::to_string(owner.mReSTIRParams.lightTileScreenSize));

    uint32_t totalCandidateCount = owner.mReSTIRParams.emissiveLightCandidateCount + owner.mReSTIRParams.envLightCandidateCount + owner.mReSTIRParams.analyticLightCandidateCount;
    float portionOfEmissiveCandidates = float(owner.mReSTIRParams.emissiveLightCandidateCount) / float(totalCandidateCount);
    float portionOfEnvironmentCandidates = float(owner.mReSTIRParams.envLightCandidateCount) / float(totalCandidateCount);
    defines.add("LIGHT_TILE_EMISSIVE_SAMPLE_COUNT", std::to_string(uint32_t(owner.mReSTIRParams.lightTileSize * portionOfEmissiveCandidates)));
    defines.add("LIGHT_TILE_ENVIRONMENT_SAMPLE_COUNT", std::to_string(uint32_t(owner.mReSTIRParams.lightTileSize * portionOfEnvironmentCandidates)));
    uint32_t lightTileAnalyticSampleCount = owner.mReSTIRParams.lightTileSize - uint32_t(owner.mReSTIRParams.lightTileSize * portionOfEmissiveCandidates) - uint32_t(owner.mReSTIRParams.lightTileSize * portionOfEnvironmentCandidates);
    defines.add("LIGHT_TILE_ANALYTIC_SAMPLE_COUNT", std::to_string(lightTileAnalyticSampleCount));

    defines.add("USE_CHECKERBOARDING", owner.mReSTIRParams.useCheckerboarding ? "1" : "0");
    defines.add("SPATIAL_VISIBILITY_THRESHOLD", std::to_string(owner.mReSTIRParams.spatialVisibilityThreshold));



    return defines;
}
