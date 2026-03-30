# GSoC 2026 Evaluation Task: SOFIE - HLS4ML Integration

This repository contains the solutions for the Google Summer of Code (GSoC) 2026 evaluation tasks for the project: "Integrating hls4ml with SOFIE for Fast ML Inference". The work demonstrates a full pipeline to parse an hls4ml in-memory ModelGraph and instantiate a ROOT SOFIE RModel for high-performance C++ inference.

## Prerequisites

To run the provided scripts, the following environment and libraries must be installed:

* ROOT (v6.30+): Built from source with the CMake flag -Dtmva-sofie=On.
* hls4ml: For the generation of the ModelGraph object.
* onnx & qonnx: For model patching and conversion to Channels-Last (NHWC) layout.
* NumPy: For contiguous memory handling and weight transposition.


## Exercise 4: The Parsing Function

The parsing function was designed to extract a complete hardware-aware configuration from an hls4ml ModelGraph.

### Implementation Details
* Dataset Focus: The algorithm was specifically optimized using the 'ConvWithAsymmetricPadding.onnx' dataset provided in the exercise. It addresses the requirement for asymmetric padding by extracting all four spatial dimensions (top, bottom, left, right), ensuring the SOFIE graph maintains mathematical parity with the original model.
* Dispatcher Pattern: The parser utilizes a modular dispatcher architecture. This allows for clean routing of layer-specific logic (Conv2D, Dense, Transpose, etc.) and makes the tool easily extensible for future operators.

## Exercise 5: The SOFIE RModel Builder

The builder script translates the parsed configuration into a functional ROOT SOFIE RModel via the Python interface.

### Implementation Details
* Technical Inspiration: The builder logic was inspired by the recent pythonization scripts of the SOFIE Keras parser found in the official ROOT repository. It follows the established pattern of registering initialized tensors before operator instantiation.
* Weight Transposition: Since hls4ml typically utilizes NHWC layout, the builder automatically transposes weights to the NCHW format required by SOFIE's optimized kernels.
* Post-Generation Patching: The script includes an automated patching mechanism to adjust the generated .hxx header, resolving class-scope shadowing and ensuring the output is ready for immediate compilation.

## File Overview

* ex4_parser.py: The modular parser for hls4ml ModelGraphs.
* ex5_builder.py: The PyROOT script to build the RModel and generate C++ code.


I affirm that I have thoroughly completed the preliminary requirements as specified:

* Exercise 1: Built ROOT from source with TMVA/SOFIE and Protobuf support, familiarizing myself with CMake and Git version control.
* Exercise 2: Gained familiarity with ROOT TMVA deep learning code, running tutorials for Higgs classification, CNN classification, and SOFIE ONNX/Keras/PyTorch inference.
* Exercise 3: Explored the hls4ml library architecture, exploring its API and the ModelGraph internal representation.
