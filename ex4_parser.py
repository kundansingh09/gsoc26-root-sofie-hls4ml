import onnx
from onnx import helper
import hls4ml
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
import qonnx.util.cleanup
from qonnx.transformation.gemm_to_matmul import GemmToMatMul 

# ==========================================
# MODULAR PARSER FUNCTIONS
# ==========================================

def extract_base_metadata(layer):
    """
    Extracts universal attributes shared across all layer types.
    Captures HLS4ML 'result_t' precision to ensure the SOFIE RModel 
    initializes with the correct ETensorType (e.g., FLOAT vs. INT).
    """
    prec_attr = layer.get_attr('result_t')
    
    if hasattr(prec_attr, 'precision'):
        precision = str(prec_attr.precision)
    elif hasattr(prec_attr, 'type'):
        precision = str(prec_attr.type)
    else:
        precision = str(prec_attr)

    base_info = {
        "type": layer.class_name,
        "input_names": getattr(layer, 'inputs', []),
        "output_names": getattr(layer, 'outputs', []),
        "reuse_factor": layer.get_attr('reuse_factor', default=1),
        "precision": precision
    }
    
    # Check for fused activations (e.g., Conv+ReLU) which SOFIE handles via AddOperator
    fused_act = layer.get_attr('activation')
    if fused_act and fused_act != 'linear':
        base_info['fused_activation'] = str(fused_act)
        
    return base_info

def parse_dense(layer, info):
    """
    Extracts weights and biases for Dense layers.
    Shape extraction is mandatory for SOFIE's AddInitializedTensor['float'] 
    which requires explicit dimension size_t vectors.
    """
    if hasattr(layer, 'get_input_variable'):
        info['input_shape'] = layer.get_input_variable().shape
    if hasattr(layer, 'get_output_variable'):
        info['output_shape'] = layer.get_output_variable().shape
        
    weight_obj = layer.get_weights('weight')
    if weight_obj:
        info['weights'] = weight_obj.data.tolist()
        info['weight_shape'] = list(weight_obj.data.shape) # Crucial for SOFIE AddInitializedTensor
        
    bias_obj = layer.get_weights('bias')
    if bias_obj:
        info['bias'] = bias_obj.data.tolist()
        info['bias_shape'] = list(bias_obj.data.shape)
        
    return info

def parse_conv(layer, info):
    """
    Extracts spatial dimensions and kernels for Convolutional layers.
    Specifically isolates padding/strides/dilations into 4-D and 2-D vectors
    to match the SOFIE ROperator_Conv C++ constructor signatures.
    """
    if hasattr(layer, 'get_input_variable'):
        info['input_shape'] = layer.get_input_variable().shape
    if hasattr(layer, 'get_output_variable'):
        info['output_shape'] = layer.get_output_variable().shape
        
    # Asymmetric padding handling: SOFIE requires [top, bottom, left, right]
    info['padding'] = [
        layer.get_attr('pad_top', default=0),
        layer.get_attr('pad_bottom', default=0),
        layer.get_attr('pad_left', default=0),
        layer.get_attr('pad_right', default=0)
    ]
    
    info['strides'] = [
        layer.get_attr('stride_height', default=1), 
        layer.get_attr('stride_width', default=1)
    ]
    info['kernel_size'] = [
        layer.get_attr('filt_height', default=1), 
        layer.get_attr('filt_width', default=1)
    ]
    
    # hls4ml often defaults these; explicit extraction ensures SOFIE graph parity
    info['dilations'] = [
        layer.get_attr('dilation_height', default=1), 
        layer.get_attr('dilation_width', default=1)
    ]
    info['groups'] = layer.get_attr('n_groups', default=1)
    
    weight_obj = layer.get_weights('weight')
    if weight_obj:
        info['weights'] = weight_obj.data.tolist()
        info['weight_shape'] = list(weight_obj.data.shape)
        # HLS4ML/ONNX uses NHWC; builder must transpose to SOFIE's NCHW requirement
        info['weight_layout'] = 'NHWC' 
        
    bias_obj = layer.get_weights('bias')
    if bias_obj:
        info['bias'] = bias_obj.data.tolist()
        info['bias_shape'] = list(bias_obj.data.shape)

    return info

def parse_pooling(layer, info):
    """Extracts window size and padding for Pooling operators."""
    if hasattr(layer, 'get_input_variable'):
        info['input_shape'] = layer.get_input_variable().shape
    if hasattr(layer, 'get_output_variable'):
        info['output_shape'] = layer.get_output_variable().shape
        
    info['pool_size'] = [
        layer.get_attr('pool_height', default=1), 
        layer.get_attr('pool_width', default=1)
    ]
    info['strides'] = [
        layer.get_attr('stride_height', default=1), 
        layer.get_attr('stride_width', default=1)
    ]
    info['padding'] = [
        layer.get_attr('pad_top', default=0),
        layer.get_attr('pad_bottom', default=0),
        layer.get_attr('pad_left', default=0),
        layer.get_attr('pad_right', default=0)
    ]
    return info

def parse_reshape(layer, info):
    """Target shape extraction for tensor re-dimensioning in SOFIE."""
    if hasattr(layer, 'get_output_variable'):
        info['target_shape'] = list(layer.get_output_variable().shape)
    return info

def parse_concat(layer, info):
    """Identifies the axis for multi-tensor join operations."""
    info['axis'] = layer.get_attr('axis', default=-1)
    return info

def parse_transpose(layer, info):
    """
    CRITICAL: Captures the permutation order to ensure axis alignment
    between the HLS4ML hardware layout and the SOFIE memory buffer.
    """
    perm = layer.get_attr('perm')
    if perm is not None:
        info['perm'] = list(perm)
    return info

def parse_activation(layer, info):
    """Handles activation parameters such as the alpha coefficient in Elu."""
    if layer.class_name == 'Elu':
        info['alpha'] = layer.get_attr('alpha', default=1.0)
    return info

# Dispatcher for modular routing of layer-specific logic
LAYER_PARSERS = {
    'Dense': parse_dense,
    'Conv1D': parse_conv,
    'Conv2D': parse_conv,
    'DepthwiseConv2D': parse_conv,
    'SeparableConv2D': parse_conv,
    'MaxPooling2D': parse_pooling,
    'AveragePooling2D': parse_pooling,
    'Reshape': parse_reshape,
    'Concatenate': parse_concat,
    'Transpose': parse_transpose,
    'Elu': parse_activation,
    'ReLU': parse_activation,
    'Input': lambda l, i: i 
}

def parse_hls_model(hls_model):
    """Main traversal function for the hls4ml ModelGraph IR."""
    model_config = {
        "graph_topology": {
            "global_inputs": [],
            "global_outputs": []
        },
        "layers": {}
    }
    
    # 1. Map Global Graph In/Out for RModel session registration
    inputs = hls_model.get_input_variables()
    outputs = hls_model.get_output_variables()
    model_config["graph_topology"]["global_inputs"] = [inp.name for inp in inputs]
    model_config["graph_topology"]["global_outputs"] = [out.name for out in outputs]

    # 2. Iterate through all ModelGraph nodes
    for layer in hls_model.get_layers():
        layer_class = layer.class_name
        
        try:
            layer_info = extract_base_metadata(layer)
            
            if layer_class in LAYER_PARSERS:
                layer_info = LAYER_PARSERS[layer_class](layer, layer_info)
            else:
                layer_info['warning'] = f"Unsupported layer {layer_class}. Base metadata only."

        except Exception as e:
            layer_info = {'type': layer_class, 'extraction_error': str(e)}
            
        model_config["layers"][layer.name] = layer_info
        
    return model_config

# ==========================================
# 1. Pre-Processing: ONNX Patching
# ==========================================
# QONNX requires strict attribute definitions. We patch the model to include explicit dilations and groups to satisfy downstream conversion validators.
print("Patching missing attributes in the ONNX file...")
original_onnx_path = '/kaggle/working/ConvWithAsymmetricPadding.onnx' 
patched_onnx_path = '/kaggle/working/exercise_model_patched.onnx'

onnx_model = onnx.load(original_onnx_path)

for node in onnx_model.graph.node:
    if node.op_type == 'Conv':
        if not any(attr.name == 'dilations' for attr in node.attribute):
            node.attribute.extend([helper.make_attribute('dilations', [1, 1])])
        if not any(attr.name == 'group' for attr in node.attribute):
            node.attribute.extend([helper.make_attribute('group', 1)])

onnx.save(onnx_model, patched_onnx_path)

# ==========================================
# 2. Transformation: QONNX format
# ==========================================
# Converts model to Channels-Last (NHWC) format, which is the native data layout for HLS4ML synthesis pipelines.
print("Converting ONNX to Channels-Last format...")
qonnx_model = ModelWrapper(patched_onnx_path)
qonnx_model = qonnx.util.cleanup.cleanup_model(qonnx_model)
qonnx_model = qonnx_model.transform(ConvertToChannelsLastAndClean())
qonnx_model = qonnx_model.transform(GemmToMatMul())
qonnx_model = qonnx.util.cleanup.cleanup_model(qonnx_model)

fixed_onnx_path = '/kaggle/working/exercise_model_fixed.onnx'
qonnx_model.save(fixed_onnx_path)

# ==========================================
# 3. Conversion: HLS4ML ModelGraph
# ==========================================
# Bridges the ONNX IR to the HLS4ML software IR.
print("Converting to hls4ml ModelGraph...")
final_onnx_model = onnx.load(fixed_onnx_path)

config = hls4ml.utils.config_from_onnx_model(
    final_onnx_model, 
    granularity='name', 
    default_precision='fixed<16,6>', 
    backend='Vitis'
)
hls_model = hls4ml.converters.convert_from_onnx_model(final_onnx_model, hls_config=config)

# ==========================================
# 4. Final Metadata Extraction
# ==========================================
print("\nParsing ModelGraph...")
final_configuration = parse_hls_model(hls_model)

print("\n=== FINAL PARSED CONFIGURATION ===")
print(f"Global Inputs: {final_configuration['graph_topology']['global_inputs']}")
print(f"Global Outputs: {final_configuration['graph_topology']['global_outputs']}")

for name, details in final_configuration["layers"].items():
    print(f"\nLayer Name: {name}")
    for k, v in details.items():
        if isinstance(v, list) and len(str(v)) > 200: 
            print(f"  {k}: <Extracted List: size {len(v)}...>")
        else:
            print(f"  {k}: {v}")