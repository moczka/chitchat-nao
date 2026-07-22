# Documentation

Uses SmolLm2-1.7B as the small language model (SLM) for NAO6 Robot.

`SmolLM2-1.7B-Instruct-Q4_K_M.gguf Q4_K_M 1.06GB`

[https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf](Download) 

---

## Commands Run

`uv python install`
`uv init`
`uv add faster-whipser`
`uv add llama-cpp-python`
`uv add webrtcvad-wheels`
`uv add pyaudio`
`uv add numpy`
`uv add --dev ruff`


## Setup Legacy Python 2.7.8

1. Download [Python 2.7.8](https://www.python.org/downloads/release/python-278/)
2. Uncompress and navigate to folder.
3. Run `OPT="" CFLAGS="-std=gnu89" ./configure --enable-optimizations`
4. Once completed run: `make && sudo make altinstall` to install.
5. Test that Python 2.7.8 installed properly with all the essential modules: `python2.7 -c "import ssl, zlib, sqlite3, readline; print('All core modules are functional!')"`
6. If you do not see the confirmation message, install the missing modules and try again by running `make clean` and starting from step #3
7.  Will need to export [Python path variable](http://doc.aldebaran.com/2-5/dev/python/install_guide.html) run `export PYTHONPATH=${PYTHONPATH}:/path/to/python-sdk/lib/python2.7/site-packages` and replace with actual absolute path to SDK
