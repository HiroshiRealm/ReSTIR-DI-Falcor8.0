from falcor import *

def render_graph_ReSTIR():
    g = RenderGraph('ReSTIR')
    ReSTIRPass = createPass('ReSTIRPass')
    g.addPass(ReSTIRPass, 'ReSTIRPass')
    VBufferRT = createPass('VBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'useTraceRayInline': False, 'useDOF': True})
    g.addPass(VBufferRT, 'VBufferRT')
    AccumulatePass = createPass('AccumulatePass', {'enabled': False, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'subFrameCount': 0, 'maxAccumulatedFrames': 0})
    g.addPass(AccumulatePass, 'AccumulatePass')
    ToneMapper = createPass('ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.addPass(ToneMapper, 'ToneMapper')
    g.addEdge('VBufferRT.vbuffer', 'ReSTIRPass.vbuffer')
    g.addEdge('VBufferRT.mvec', 'ReSTIRPass.mvec')
    g.addEdge('ReSTIRPass.color', 'AccumulatePass.input')
    g.addEdge('AccumulatePass.output', 'ToneMapper.src')
    g.markOutput('ToneMapper.dst')
    return g

ReSTIR = render_graph_ReSTIR()
try: m.addGraph(ReSTIR)
except NameError: None
