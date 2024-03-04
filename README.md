This is an experiment to use WebGPU Compute Shaders to decode JPEG images.
The main Purpose of this project is to learn how to write compute shaders.

It performs surprisingly well on my RX 6700 XT, with GPU times of around 1 ms
at high GPU clocks (and 2ms spent doing preprocessing on the CPU) when decoding
a 4k test image.

The approach used here is restricted to baseline JPEGs that make heavy use
of restart intervals. These types of JPEGs are typically produced by hardware
encoders in GPUs, phones and webcams.

Due to technical limitations, only YUV JPEGs that make use of 4:2:2 chroma
subsampling are supported. In the future this restriction may be lifted.
