# Blazing-fast CPU inference for multilingual-e5-small (4-bit, pure C)
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -funroll-loops -ffast-math -fopenmp -Wall -Wextra
LDFLAGS ?= -lm -fopenmp -lpthread

MODEL   ?= e5-small-q4.bin

.PHONY: all lib cli server convert clean test bench stress

all: lib server

# shared library used by the Python ctypes wrapper
lib: libe5.so
libe5.so: e5.c e5.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ e5.c $(LDFLAGS)

# self-contained server+CLI binary with the 4-bit model embedded inside it.
# A client only needs this single file -- no model, no Python, no deps.
server: e5
e5: e5.c e5.h server.c model_embed.o
	$(CC) $(CFLAGS) -DE5_EMBED -o $@ e5.c server.c model_embed.o $(LDFLAGS)

# inject the model file into a linkable object (symbols _binary_model_bin_*)
model_embed.o: $(MODEL)
	cp -f $(MODEL) model.bin
	ld -r -b binary -o $@ model.bin
	rm -f model.bin

# tiny standalone CLI that loads the model from a file (no embedding)
cli: e5cli
e5cli: e5.c e5.h
	$(CC) $(CFLAGS) -DE5_MAIN -o $@ e5.c $(LDFLAGS)

# build the 4-bit model file from the HF checkpoint
convert: $(MODEL)
$(MODEL): convert.py
	python3 convert.py

test: lib $(MODEL)
	python3 test_parity.py

stress: lib e5
	python3 stress_test.py

bench: e5
	./e5 query "how much protein should a female eat"

clean:
	rm -f libe5.so e5cli e5 model_embed.o model.bin
