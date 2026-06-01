#!/usr/bin/env python3

from graphviz import Digraph
import subprocess
import os

VIRGIL_FONT_PATH = os.environ.get("VIRGIL_FONT_PATH", "Virgil")


def log_font_info():
    print(f"[INFO] Setting fontname to: {VIRGIL_FONT_PATH}")
    try:
        print("[INFO] Checking for Virgil font in system fonts...")
        fc_list = subprocess.check_output(["fc-list", ":", "family,file"]).decode()
        found = False
        for line in fc_list.splitlines():
            if VIRGIL_FONT_PATH in line or "Virgil" in line:
                print(f"[FOUND] {line}")
                found = True
        if not found:
            print(
                f"[WARNING] Virgil font not found in system font list or at {VIRGIL_FONT_PATH}!"
            )
        else:
            print("[INFO] Virgil font is available to the system.")
    except Exception as e:
        print(f"[ERROR] Could not check system fonts: {e}")
    print(
        "[INFO] If the font is not found, make sure fg-virgil is installed and font cache is updated."
    )


def create_llm_bottlenecks_diagram():
    dot = Digraph(comment="LLM Inference vs Training Bottlenecks")
    dot.attr(rankdir="TB")  # Top to bottom layout
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Virgil",
        fontpath=VIRGIL_FONT_PATH,
    )
    dot.attr("edge", color="gray50", fontname=VIRGIL_FONT_PATH)

    # Inference Mode Cluster
    with dot.subgraph(name="cluster_inference") as c:
        c.attr(label="Inference Mode (Single User)")
        c.attr(style="rounded,filled")
        c.attr(fillcolor="lightblue")
        c.attr(fontsize="16")
        
        # Main components
        c.node("user", "Single User\nRequest", fillcolor="lightgreen")
        c.node("gpu_inf", "GPU\n(High Memory Bandwidth Required)", fillcolor="lightpink")
        c.node("memory_inf", "Memory\n(Bottleneck: 1TB/s Bandwidth)", fillcolor="lightyellow")
        
        # Connections
        c.edge("user", "gpu_inf", label="Low Latency\nRequired")
        c.edge("gpu_inf", "memory_inf", label="Memory\nBound")
        
        # Bottleneck indicator
        c.node("bottleneck_inf", "Bottleneck:\nMemory Bandwidth", fillcolor="red", shape="diamond")
        c.edge("memory_inf", "bottleneck_inf", style="dashed", color="red")

    # Training Mode Cluster
    with dot.subgraph(name="cluster_training") as c:
        c.attr(label="Training Mode (Bulk Processing)")
        c.attr(style="rounded,filled")
        c.attr(fillcolor="lightblue")
        c.attr(fontsize="16")
        
        # Main components
        c.node("data", "Large Dataset\n(Millions of Tokens)", fillcolor="lightgreen")
        c.node("gpu_train", "GPU\n(High Compute Required)", fillcolor="lightpink")
        c.node("compute_train", "Compute\n(Bottleneck: 312 TFLOPS)", fillcolor="lightyellow")
        c.node("network_train", "Network I/O\n(100 Gbps InfiniBand)", fillcolor="lightcyan")
        
        # Connections
        c.edge("data", "gpu_train", label="Bulk Processing")
        c.edge("gpu_train", "compute_train", label="Compute\nBound")
        c.edge("gpu_train", "network_train", label="Network\nOverlap")
        
        # Bottleneck indicator
        c.node("bottleneck_train", "Bottleneck:\nGPU FLOPS", fillcolor="red", shape="diamond")
        c.edge("compute_train", "bottleneck_train", style="dashed", color="red")

    # Add a connecting line between the two modes
    dot.edge("bottleneck_inf", "bottleneck_train", 
             label="Different Optimization Goals:\nLatency vs Throughput", 
             style="solid", 
             color="gray30",
             constraint="false",
             dir="none")  # Remove arrow

    dot.attr(
        label="LLM Inference vs Training: Different Bottlenecks\n(Simplified Architecture)",
        fontname=VIRGIL_FONT_PATH,
    )
    dot.attr(fontsize="20")
    return dot


def main():
    log_font_info()
    dot = create_llm_bottlenecks_diagram()
    # Save the diagram to assets directory
    output_path = "assets/llm-bottlenecks"
    dot.render(output_path, format="png", cleanup=True)
    print(f"[INFO] Diagram saved as '{output_path}.png'")
    return dot


if __name__ == "__main__":
    main() 