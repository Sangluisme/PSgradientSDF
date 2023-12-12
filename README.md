## <b>High-Quality RGB-D Reconstruction via Multi-View Uncalibrated Photometric Stereo and Gradient-SDF</b>

WACV 2023 [paper](https://arxiv.org/abs/2210.12202)
---
Enable detailed RGB-D data 3D reconstruction. Jointly estimate camera pose, geometry, albedo and environment lighting under natural light or point-light-source.

![method pipeline](pipeline.png)

# clone repository
```
git clone https://github.com/Sangluisme/PSgradientSDF.git
```
please use
```
git submodule update --init --recursive
```
to pull our dependency.

# code structure
``` 
cpp (basic code folder)
|---include
|---third
|---voxel_ps
|     |---bin
|     |---src
|     |---CMakeLsits.txt
|---CMakeLists.txt

config (example config files)
|---config.json
|---...

data (demo data folder)
|---demo_data1
|---demo_data2

results (default results save path)
|---...

```
# setup
please create a `build` folder under the [cpp/](cpp/) folder path then do 
```
cd cuild
cmake ..
make 
```
it will build a binary file under the path [cpp/voxel_ps/bin/](cpp/voxel_ps/bin/).

# usage
```
cd ./cpp/voxel_ps/bin/
./voxelPS --config_file <your config file path>
```

# config file
All the parameters you can control are listed in the example config files in the [config](config/) folder. 
Here is some explanation of the parameters.

- **input**: data folder path
- **output**: results saving path
- pose filename: ground true pose or pre-calculated pose file name under input path (optional)
- **datatype**: support `tum` (TUM_RGBD sequence), `synth`, `multiview` differences explained in next section
- first: start frame number
- last: end frame number
- voxel size: voxel size in `m` (choose a larger one if the algorithm is too slow)
- sharpness threshold: criteria of selecting the keyframe
- **model type**: `SH1`, `SH2`(not recommend), `LED` (1st, 2nd spherical harmonics, point-light-source)
- reg albedo: regularizer for albedo (set to 0)
- reg norm: Eikonal regularizer
- reg laplacian: regularizer for distance laplacian (set to 0)
- lambda: loss function parameters
- --light: bool if update light
- --albedo: bool if update albedo
- --distance: bool if update SDF distance
- --pose: bool if update pose

please note the **bold** parameters are required, others are optional.

# data type
**The main difference between each data type is the structure of the data folder**
The general requirement is:
- contain depth and RGB images, together with `intrinsics.txt` in the same folder. 
- If the pose is unknown, the images should be like a video sequence that allow camera tracking, otherwise initial camera pose file should be provided. 

`tum` -- TUM_RGBD (please refer to https://vision.in.tum.de/data/datasets/rgbd-dataset/download for detail)

should have a structure 
```
data
|---depth (folder)
|     |---depth_timestamp1.png
|     |---depth_timestamp2.png
|     |---...
|
|---rgb (folder)
|     |---rgb_timestamp1.png
|     |---rgb_timestamp2.png
|     |---...
|
|---depth.txt
|---rgb.txt
|---associated.txt
|---intrinsics.txt

```
`multiview` -- intrinsic3d data should have the structure (some multi view data has too large baseline, thus initial poses are needed)
```
data
|---color0000001.png
|---color0000002.png
|---color0000003.png
|---...
|---depth0000001.png
|---depth0000001.png
|---depth0000001.png
|---...
|---intrinsics.txt
|---pose.txt 
```
`synth` -- the synthetic data or point-light-source data which is recorded using the set-up mentioned in the paper

```
data
|---depth
|     |---001.png
|     |---002.png
|     |---003.png
|     |---...
|---rgb
|     |---001.png
|     |---003.png
|     |---003.png
|     |---...
|---intrinsics.txt
```
**To use your own data, just convert your data to either one of the structures and specify the corresponding data type in the `config.json` file**.

# trouble shooting

- **compile error of `Sophus`**: we use an older version of sophus, just commit back to the version shows in the git repository.
- **too slow/out of memory**: disable `upsampling` in `config.json` or increase the voxel size.
- **reconstruction size**: it is controlled by two factors, voxel size (per voxel, in meters) and voxel grid size, the default is 128x128x128. The actual reconstruction range will be voxel size times voxel grid size, e.g. 0.02*128 = 2.56 m (one edge). If you would like to use a bigger voxel grid size, change it here https://github.com/Sangluisme/PSgradientSDF/blob/164869288ffa2cca0162e79e262614d9309da57d/cpp/voxel_ps/src/main_ps.cpp#L123.
- **voxel size**: changing voxel size will influence the reconstruction details since a smaller voxel size means each voxel only represents a smaller area. This will not affect the memory but a smaller voxel size means under the same voxel grid size, the reconstruction area will be small.
- **voxel grid size**: currently is hard coded as 128x128x128. Please change here if you want https://github.com/Sangluisme/PSgradientSDF/blob/164869288ffa2cca0162e79e262614d9309da57d/cpp/voxel_ps/src/main_ps.cpp#L123. Larger voxel grid size will cause large memory consumptions, but also allows a larger reconstruction range.

# citation
```
@inproceedings{sang2023high,
 author = {L Sang and B Haefner and X Zuo and D Cremers},
 title = {High-Quality RGB-D Reconstruction via Multi-View Uncalibrated Photometric Stereo and Gradient-SDF},
 booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
 month = {January},
 address = {Hawaii, USA},
 year = {2023},
 eprint = {2210.12202},
 eprinttype = {arXiv},
 eprintclass = {cs.CV},
 copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International},
 keywords = {3d-reconstruction,rgb-d,photometry},
}
```
