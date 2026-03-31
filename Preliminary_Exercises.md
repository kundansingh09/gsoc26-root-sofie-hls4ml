# Preliminary Tasks Documentation

**Project:** Integrating hls4ml with SOFIE for Fast ML Inference

## Exercise 1: Building ROOT from Source

**Objective:** Build ROOT with TMVA/SOFIE and Protobuf support using CMake, and establish version control.

### 1. Environment Setup & Dependencies
Ensured all required system libraries (`cmake`, `protobuf-compiler`, `libprotobuf-dev`) and Python dependencies (`tensorflow`, `torch`, `hls4ml`) were installed.

### 2. Version Control Setup
Cloned the repository and created a dedicated working branch:

```bash
git clone [https://github.com/kundansingh09/ML4EP-root.git](https://github.com/kundansingh09/ML4EP-root.git) root_src
cd root_src
git checkout -b my-sofie-dev-branch
```

### 3. Configuring the Build
Created an out-of-source build directory and configured CMake with Ninja, explicitly enabling TMVA, SOFIE, and PyROOT:

```bash
mkdir -p root_build root_install
cd root_build
cmake ../root_src -G Ninja -Dtmva-sofie=On -Dpyroot=ON -Dtmva=On -Dbuiltin_lz4=ON -DCMAKE_INSTALL_PREFIX=../root_install
```
*Build Configuration Output Snippet:*

```text
-- Enabled support for: ... pyroot ... tmva tmva-sofie ...
-- Configuring done
-- Generating done
-- Build files have been written to: /tmp/root_build
```

### 4. Building and Installing
Compiled the source and installed the binaries:

```bash
cmake --build . -j8 
cmake --install .
```



### 5. Build Verification
Executed a Python script to verify the newly built ROOT environment and confirm the SOFIE module was active:

```python
import sys
import ROOT

sys.path.append("/tmp/root_install/lib")

print(f"ROOT Version: {ROOT.gROOT.GetVersion()}")
if hasattr(ROOT, "TMVA") and hasattr(ROOT.TMVA.Experimental, "SOFIE"):
    print("TMVA SOFIE is active and ready!")
else:
    print("SOFIE module not found.")
```
*Verification Output:*

```text
ROOT successfully imported!
ROOT Version: 6.39.01
TMVA SOFIE is active and ready for Exercise 4/5!
```
