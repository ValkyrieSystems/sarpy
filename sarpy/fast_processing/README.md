# fast_processing

The `fast_processing` subpackage contains faster, but less flexible implementations of common processing functions.


## sicd_to_detected_image
The `sicd_to_detected_image` utility can be used to produce a SIDD NITF file from a SICD file.

```
python -m sarpy.fast_processing.sicd_to_detected_image  <path_to_input_sicd> <path_to_output_sidd>
```
## adjust_sicd_osr
The `adjust_sicd_osr` utility can be used to produce a SICD NITF file with a modified sampling rate from a SICD file.

```
python -m sarpy.fast_processing.adjust_sicd_osr --desired-osr <osr> <path_to_input_sicd> <path_to_output_sicd>
```
## sidelobe_control
The `sidelobe_control` utility can be used to produce a SICD NITF with modified sidelobe control from a SICD file.

```
python -m sarpy.fast_processing.sidelobe_control --sidelobe-control <type> <path_to_input_sicd> <path_to_output_sicd>
```
