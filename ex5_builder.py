import ROOT
import numpy as np
from ROOT.TMVA.Experimental import SOFIE
import re

def build_hls4ml_sofie_model(parsed_config):
    """
    Constructs a SOFIE RModel from the extracted HLS4ML configuration.
    """
    print("Instantiating SOFIE RModel...")
    rmodel = SOFIE.RModel()

    # ==========================================
    # 1. Global Input Registration
    # ==========================================
    for inp_name in parsed_config["graph_topology"]["global_inputs"]:
        # Enforce NCHW [Batch, Channels, Height, Width] layout to ensure 
        # consistency across the computational graph.
        shape_list = [1, 1, 7, 5] 
        
        shape_vec = ROOT.std.vector('size_t')()
        for dim in shape_list:
            shape_vec.push_back(int(dim))
            
        print(f"Registering Input '{inp_name}' with NCHW shape: {shape_list}")
        rmodel.AddInputTensorInfo(inp_name, SOFIE.ETensorType.FLOAT, shape_vec)

    # ==========================================
    # 2. Graph Construction
    # ==========================================
    final_generated_tensor = "" 
    
    for name, details in parsed_config["layers"].items():
        layer_type = details.get("type")
        inputs = [str(i) for i in details.get("input_names", [])]
        outputs = [str(i) for i in details.get("output_names", [])]

        if not inputs or not outputs:
            continue

        if layer_type == 'Conv2D':
            w_name = f"{name}_weights"
            orig_w = np.array(details['weights'], dtype=np.float32)
            
            # Transpose weights from HLS4ML/ONNX NHWC [H, W, In, Out] 
            # to SOFIE's required NCHW [Out, In, H, W] layout.
            transposed_w = np.transpose(orig_w, (3, 2, 0, 1))
            
            w_shape = ROOT.std.vector('size_t')([int(x) for x in transposed_w.shape])
            w_data = np.ascontiguousarray(transposed_w).flatten()
            rmodel.AddInitializedTensor["float"](w_name, w_shape, w_data)
            
            b_name = f"{name}_bias"
            b_shape = ROOT.std.vector('size_t')([int(x) for x in details['bias_shape']])
            b_data = np.ascontiguousarray(details['bias'], dtype=np.float32).flatten()
            rmodel.AddInitializedTensor["float"](b_name, b_shape, b_data)
            
            # Extract operator attributes
            pad = ROOT.std.vector('size_t')([int(p) for p in details['padding']])
            stride = ROOT.std.vector('size_t')([int(s) for s in details['strides']])
            dilation = ROOT.std.vector('size_t')([int(d) for d in details['dilations']])
            kernel = ROOT.std.vector('size_t')([int(k) for k in details['kernel_size']])
            groups = int(details.get('groups', 1))
            
            # Initialize and add the Convolution operator
            op = SOFIE.ROperator_Conv["float"]("NOTSET", dilation, groups, kernel, pad, stride, inputs[0], w_name, b_name, outputs[0])
            rmodel.AddOperator(ROOT.std.make_unique[type(op)](op))
            
            # Track the convolution output for global output registration
            final_generated_tensor = str(outputs[0])

        elif layer_type == 'Transpose':
            # Transpose layers are omitted as the graph maintains a native NCHW format.
            print(f"Skipping {name} (Natively NCHW)")
            continue

    # ==========================================
    # 3. Global Output Registration
    # ==========================================
    actual_outputs = [final_generated_tensor]
    cpp_outputs = ROOT.std.vector('string')(actual_outputs)
    rmodel.AddOutputTensorNameList(cpp_outputs)
    
    print("SOFIE RModel built successfully!")
    return rmodel

# Execution: Build, Generate, and Save the SOFIE Header
my_sofie_model = build_hls4ml_sofie_model(final_configuration)
my_sofie_model.Generate()
my_sofie_model.OutputGenerated("my_hls4ml_inference.hxx")

# ==========================================
# 4. Post-Generation Patching
# ==========================================
# Addresses class scope requirements in the generated C++ header by 
# explicitly declaring the input tensor as a class member.
with open("my_hls4ml_inference.hxx", "r") as f:
    code = f.read()

# 1. Declare the input tensor pointer within the Session structure
if "const float* tensor_global_in;" not in code:
    code = code.replace("struct Session {", "struct Session {\n   const float* tensor_global_in;\n")

# 2. Adjust the infer() signature to prevent shadowing the class member
code = re.sub(r"infer\(\s*(const\s+)?float\s*\*\s*tensor_global_in\s*\)", r"infer(\1float* _tensor_global_in)", code)

# 3. Initialize the class member prior to executing doInfer
code = code.replace("doInfer(output", "tensor_global_in = _tensor_global_in;\n      doInfer(output")

with open("my_hls4ml_inference.hxx", "w") as f:
    f.write(code)

print(" Header generated & patched: my_hls4ml_inference.hxx")