# Documentation

Uses SmolLm2-1.7B as the small language model (SLM) for NAO6 Robot.

`SmolLM2-1.7B-Instruct-Q4_K_M.gguf Q4_K_M 1.06GB`

[https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf](Download) 

---

## NAO6 Robot SDK

This project uses the (NAOqi 2.8)[http://doc.aldebaran.com/2-8/dev/naoqi/index.html] which is currently the latest version and the
version used in this project. This version offers Python3 C++ bindings through the [LibQi](https://pypi.org/project/qi/) "qi" Python module.
For C++, we use [naoqi-sdk-2.8.5.10](https://maxtronics.com/en/support/kb/nao6/downloads/nao6-software-downloads/) which will need to be downloaded
and referenced in the CMake setup file.

## NAOqi Framework

It is important to understand the architecture of the NAOqi framework in order to implement its APIs correctly. This (introduction documentation)[http://doc.aldebaran.com/2-8/dev/naoqi/index.html] explains the architecture behind the framework and how to work with its APIs via the core modules provided. Make sure to read the high level (introduction)[http://doc.aldebaran.com/2-8/naoqi/core/index.html#naoqi-core] to all the core modules, additional modules for motion, audio and vision are documented in more (detailed here)[http://doc.aldebaran.com/2-8/naoqi/index.html]. There you will be able to learn everything about each module and its APIs as well as view code examples.

## Commands Run

`uv sync`



