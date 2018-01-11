
all:
	g++ extractFrames.cpp `pkg-config --libs --cflags opencv` -o extractFrames

convert:
	g++ convertPreScanToSynthia.cpp `pkg-config --libs --cflags opencv` -o convertLabel


convert21:
	g++ convertPreScanToCityscapes.cpp `pkg-config --libs --cflags opencv` -o convertLabel21
