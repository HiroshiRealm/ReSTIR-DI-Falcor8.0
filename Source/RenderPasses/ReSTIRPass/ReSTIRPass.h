/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "Falcor.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Utils/Sampling/AliasTable.h"
#include "RenderGraph/RenderPassHelpers.h"

using namespace Falcor;

/**
 * ReSTIR Direct Illumination render pass.
 *
 * This pass implements the ReSTIR DI (Reservoir-based Spatiotemporal Importance Resampling
 * for Direct Illumination) algorithm for real-time ray tracing.
 */
class ReSTIRPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReSTIRPass, "ReSTIRPass", "ReSTIR Direct Illumination pass.");

    static ref<ReSTIRPass> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<ReSTIRPass>(pDevice, props);
    }

    ReSTIRPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    enum class Mode
    {
        NoResampling,
        SpatialResampling,
        TemporalResampling,
        SpatiotemporalResampling,
        DecoupledPipeline,
        ReSTIRGI,
    };

    enum class BiasCorrection
    {
        Off,
        Naive,
        MIS,
        RayTraced,
    };

private:

    struct TracePass
    {
        std::string name;                           ///< Name of the TracePass.
        std::string passDefine;                     ///< Definition string associated with the TracePass.
        ref<Program> pProgram;                      ///< Shared pointer to the raytracing program for this pass.
        ref<RtBindingTable> pBindingTable;         ///< Shared pointer to the binding table for this pass.
        ref<RtProgramVars> pVars;                  ///< Shared pointer to the program variables for this pass.

        TracePass(ref<Device> pDevice, const std::string& name, const std::string& passDefine, const ref<Scene>& pScene, const DefineList& defines, const TypeConformanceList& globalTypeConformances);
        void prepareProgram(ref<Device> pDevice, const DefineList& defines);    ///< Prepares the raytracing program for this pass.
    };

    ReSTIRPass(const Properties& props);

    void parseProperties(const Properties& props);

    void prepareMaterials(RenderContext* pRenderContext);

    bool prepareLighting(RenderContext* pRenderContext);

    void resetLighting();

    std::unique_ptr<AliasTable> createEmissiveGeometryAliasTable(RenderContext* pRenderContext, const ref<LightCollection>& lightCollection);
    std::unique_ptr<AliasTable> createEnvironmentAliasTable(RenderContext* pRenderContext, const ref<Texture>& envTexture);
    std::unique_ptr<AliasTable> createAnalyticLightsAliasTable(RenderContext* pRenderContext);

    bool beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void endFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void setFrameDim(const uint2 frameDim);
    void updatePrograms();
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);

    /*
     * Render passes.
     */
    void tracePass(RenderContext* pRenderContext, const RenderData& renderData, TracePass& tracePass);
    void createLightTiles(RenderContext* pRenderContext);
    void loadSurfaceDataPass(RenderContext* pRenderContext, const RenderData& renderData);
    void generateInitialCandidatesPass(RenderContext* pRenderContext, const RenderData& renderData);
    void temporalReusePass(RenderContext* pRenderContext, const RenderData& renderData);
    void spatialReusePass(RenderContext* pRenderContext, const RenderData& renderData);
    void createDirectSamplesPass(RenderContext* pRenderContext, const RenderData& renderData);
    void shadePass(RenderContext* pRenderContext, const RenderData& renderData);
    void temporalReuseGIPass(RenderContext* pRenderContext, const RenderData& renderData);
    void spatialReuseGIPass(RenderContext* pRenderContext, const RenderData& renderData);
    void shadingIndirectPass(RenderContext* pRenderContext, const RenderData& renderData);
    void decoupledPipelinePass(RenderContext* pRenderContext, const RenderData& renderData);

    void prepareRenderPass(const RenderData& renderData);
    void setShaderData(const ShaderVar& var, const RenderData& renderData, bool useLightSampling = true) const;
    bool renderRenderingUI(Gui::Widgets& widget);

    /** Static configuration. Changing any of these options require shader recompilation.
    */
    struct StaticParams
    {
        // Sampling parameters
        uint32_t    sampleGenerator = SAMPLE_GENERATOR_TINY_UNIFORM; ///< Pseudorandom sample generator type.
        EmissiveLightSamplerType emissiveSampler = EmissiveLightSamplerType::Power;  ///< Emissive light sampler to use.

        DefineList getDefines(const ReSTIRPass& owner) const;
    };

    /*
     * Static configuration of the ReSTIR algorithm.
     */
    struct ReSTIRParams
    {
        uint32_t    lightTileScreenSize = 8;                    ///< Screen size of the light tiles in pixels.
        uint32_t    lightTileSize = 1024;                       ///< Total number of light samples in each light tile.
        uint32_t    lightTileCount = 128;                       ///< Total number of light tiles.

        bool        testInitialSampleVisibility = true;         ///< If true, initial samples' visibility is tested.
        uint32_t    emissiveLightCandidateCount = 24;    ///< Number of candidate samples for emissive lights.
        uint32_t    envLightCandidateCount = 8;           ///< Number of candidate samples for environment lights.
        uint32_t    analyticLightCandidateCount = 1;      ///< Number of candidate samples for analytic lights.

        BiasCorrection biasCorrection = BiasCorrection::Off;    ///< Bias correction method used.
        float       normalThreshold = 0.9f;                     ///< Threshold for normal comparison.
        float       depthThreshold = 0.1f;                      ///< Threshold for depth comparison.

        float       emissiveIntensityMultiplier = 30.0f;         ///< Multiplier for emissive light intensity to make them more visible.

        uint32_t    spatialIterationCount = 1;                  ///< Number of spatial resampling iterations.
        uint32_t    spatialReuseSampleCount = 5;                ///< Number of samples reused from the previous frame.
        float       spatialReuseSampleRadius = 50.f;            ///< Radius within which to reuse samples.

        uint32_t    temporalHistoryLength = 20;                 ///< Length of the temporal history for resampling.

        bool        useCheckerboarding = false;                 ///< If true, checkerboard rendering is used.

        float       spatialVisibilityThreshold = 0.f;           ///< Threshold for visibility during spatial resampling.

        Mode        mode = Mode::SpatiotemporalResampling;      ///< The resampling mode of ReSTIR algorithm.


    };

    // Configuration
    StaticParams                    mStaticParams;              ///< Static parameters. These are set as compile-time constants in the shaders.
    bool                            mEnabled = true;            ///< Switch to enable/disable the render pass. When disabled the pass outputs are cleared.
    RenderPassHelpers::IOSize       mOutputSizeSelection = RenderPassHelpers::IOSize::Default;  ///< Selected output size.
    uint2                           mFixedOutputSize = { 512, 512 };                      ///< Output size in pixels when 'Fixed' size is selected.


    ReSTIRParams                    mReSTIRParams;              ///< Contains parameters for the ReSTIR algorithm.

    // Internal state
    ref<Scene>                      mpScene;                    ///< The current scene, or nullptr if no scene loaded.
    ref<SampleGenerator>            mpSampleGenerator;          ///< GPU pseudo-random sample generator.
    ref<EnvMapSampler>              mpEnvMapSampler;            ///< Environment map sampler or nullptr if not used.
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler;    ///< Emissive light sampler or nullptr if not used.

    ref<ComputePass>                mpCreateLightTiles;                 ///< Compute pass for creating light tiles.
    ref<ComputePass>                mpLoadSurfaceDataPass;              ///< Compute pass for loading surface data.
    ref<ComputePass>                mpGenerateInitialCandidatesPass;    ///< Compute pass for generating initial candidates.
    ref<ComputePass>                mpTemporalReusePass;                ///< Compute pass for temporal reuse.
    ref<ComputePass>                mpSpatialReusePass;                 ///< Compute pass for spatial reuse.
    ref<ComputePass>                mpCreateDirectLightSamplesPass;     ///< Compute pass for creating direct light samples.
    ref<ComputePass>                mpShadePass;                        ///< Compute pass for shading.
    ref<ComputePass>                mpShadingIndirect;                  ///< Compute pass for indirect shading.
    ref<ComputePass>                mpTemporalReuseGIPass;              ///< Compute pass for temporal reuse in global illumination.
    ref<ComputePass>                mpSpatialReuseGIPass;               ///< Compute pass for spatial reuse in global illumination.

    ref<ComputePass>                mpDecoupledPipelinePass;            ///< Compute pass for decoupled pipeline.

    std::unique_ptr<TracePass>      mpTracePass;                        ///< Main trace pass.

    // Runtime data
    uint                            mFrameCount = 0;                    ///< Frame count since scene was loaded.
    uint2                           mFrameDim = uint2(0, 0);      ///< Dimensions of the current frame.

    bool                            mOptionsChanged = false;    ///< Flag indicating whether the options have changed.
    bool                            mRecompile = false;         ///< Set to true when program specialization has changed.
    bool                            mVarsChanged = true;        ///< This is set to true whenever the program vars have changed and resources need to be rebound.
    bool                            mClearReservoirs = false;   ///< Set to true when reservoir history should be cleared due to lighting changes.

    // Textures and buffer
    ref<Buffer> mpReservoirs;                     ///< Pointer to the buffer for reservoirs.
    ref<Buffer> mpDirectLightSamples;             ///< Pointer to the buffer for direct light samples.
    ref<Buffer> mpSurfaceData;                    ///< Pointer to the buffer for surface data.
    ref<Buffer> mpNormalDepth;                    ///< Pointer to the buffer for normal depth.

    ref<Buffer> mpPrevSurfaceData;                ///< Pointer to the buffer for previous surface data.
    ref<Buffer> mpPrevNormalDepth;                ///< Pointer to the buffer for previous normal depth.
    ref<Buffer> mpPrevReservoirs;                 ///< Pointer to the buffer for previous reservoirs.

    ref<Buffer> mpGIReservoirs;                   ///< Pointer to the buffer for initial global illumination reservoirs.
    ref<Buffer> mpPrevGIReservoirs;               ///< Pointer to the buffer for previous global illumination reservoirs.
    ref<Buffer> mpSpatialGIReservoirs;            ///< Pointer to the buffer for spatial global illumination reservoirs.

    // Emissive geometry sampling data
    std::unique_ptr<AliasTable> mpEmissiveGeometryAliasTable; ///< Pointer to the alias table for emissive geometry.
    std::unique_ptr<AliasTable> mpEnvironmentAliasTable;      ///< Pointer to the alias table for environment.
    std::unique_ptr<AliasTable> mpAnalyticLightsAliasTable;   ///< Pointer to the alias table for analytic lights.

    ref<Buffer> mpEnvironmentLuminanceTable;      ///< Pointer to the buffer for environment luminance table.

    ref<Buffer> mpLightTiles;                     ///< Pointer to the buffer for light tiles.

    std::mt19937 mRnd;                                  ///< Random number generator.
};
