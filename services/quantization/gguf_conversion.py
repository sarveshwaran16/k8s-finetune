import os
import subprocess
import sys

MERGED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "k8s-tinyllama-merged")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "k8s-tinyllama-q4.gguf")

# Downloaded prebuilt binary -- update this if you move the file.
LLAMA_QUANTIZE_EXE = r"D:\rag-assistant\k8s-finetune\llama-cpp-bin\llama-quantize.exe"


def run():
    # Step 1 — clone llama.cpp if not already present (needed for the
    # Python conversion script, NOT for llama-quantize itself since
    # that's already a prebuilt binary now)
    llama_cpp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "llama.cpp")
    if not os.path.exists(llama_cpp_dir):
        print("Cloning llama.cpp...")
        subprocess.run([
            "git", "clone",
            "https://github.com/ggerganov/llama.cpp.git",
            llama_cpp_dir
        ], check=True)
    else:
        print("llama.cpp already cloned, skipping...")

    # Step 2 — install llama.cpp Python requirements
    req_file = os.path.join(llama_cpp_dir, "requirements.txt")
    if os.path.exists(req_file):
        print("Installing llama.cpp requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file, "-q"], check=True)

    # Step 3 — convert to GGUF (fp16 first)
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    fp16_output = OUTPUT_PATH.replace("q4.gguf", "fp16.gguf")
    print(f"Converting to GGUF (fp16)...")
    subprocess.run([
        sys.executable, convert_script,
        MERGED_MODEL_PATH,
        "--outtype", "f16",
        "--outfile", fp16_output
    ], check=True)
    print(f"✅ GGUF (fp16) saved to {fp16_output}")

    # Step 4 — quantize to Q4_K_M using the downloaded prebuilt binary
    if not os.path.exists(LLAMA_QUANTIZE_EXE):
        print(f"❌ llama-quantize.exe not found at {LLAMA_QUANTIZE_EXE}")
        print("Double-check the path, or update LLAMA_QUANTIZE_EXE at the top of this script.")
        print(f"fp16 GGUF is at: {fp16_output} (usable directly with Ollama if quantization is skipped)")
        return

    print("Quantizing to Q4_K_M...")
    subprocess.run([
        LLAMA_QUANTIZE_EXE,
        fp16_output,
        OUTPUT_PATH,
        "Q4_K_M"
    ], check=True)
    print(f"✅ Quantized model saved to {OUTPUT_PATH}")

    # Report final file size -- this is your actual PRD "<300MB GGUF" number
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Final quantized size: {size_mb:.1f} MB (PRD target: <300MB)")


if __name__ == "__main__":
    run()