This is an experiment to use WebGPU Compute Shaders to decode JPEG images.

It works remarkably well on my RX 6700 XT, with frame times of around 13ms
(2ms spent doing preprocessing on the CPU, around 10ms spent on the GPU). This
beats all of mozjpeg, jpeg-decoder, and zune-jpeg. Even a very multithreaded and
very specifically patched version of zune-jpeg only manages 17ms frame times on
my workstation.

Unfortunately, Intel's iGPUs are, apparently, 10-20x worse at this, so the
machine I wanted to use this on ended up with 100-200ms frame times, which,
to put it mildly, is "not very good".

The quest for faster JPEGs continues.
