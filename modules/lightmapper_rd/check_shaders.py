#!/usr/bin/env python3
"""Compiles the lightmapper's shaders with glslang, without the editor.

Expands each shader file the way Godot's RDShaderFile does: every #[versions] entry (or none),
each stage section (#[compute], #[vertex], #[fragment]), #VERSION_DEFINES replaced by the version's
define plus the bake options LightmapperRD adds (all off, and all on), and includes inlined. Prints
glslang's errors with the variant they came from. Exit status is nonzero if any variant fails.

Usage: check_shaders.py [glslang binary] (default: ~/.local/opt/glslang/bin/glslang or `glslang` on
the PATH).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SHADERS = ["lm_compute.glsl", "lm_raster.glsl", "lm_blendseams.glsl"]
# What LightmapperRD::bake adds to every variant, with the optional bake options off and on.
OPTION_SETS = [
    "#define CLUSTER_SIZE 16\n",
    "#define CLUSTER_SIZE 16\n#define USE_SH_LIGHTMAPS\n#define USE_LIGHT_TEXTURE_FOR_BOUNCES\n#define USE_SHADOWMASK\n",
]
STAGES = {"compute": "comp", "vertex": "vert", "fragment": "frag"}


def find_glslang():
    if len(sys.argv) > 1:
        return sys.argv[1]
    local = os.path.expanduser("~/.local/opt/glslang/bin/glslang")
    return local if os.path.exists(local) else shutil.which("glslang")


def inline_includes(text, base_dir, seen=()):
    def replace(match):
        path = os.path.join(base_dir, match.group(1))
        if path in seen:
            return ""
        with open(path) as f:
            return inline_includes(f.read(), os.path.dirname(path), seen + (path,))

    return re.sub(r'^#include "([^"]+)"', replace, text, flags=re.M)


def parse(path):
    """Returns ({version name: define}, {stage: source})."""
    with open(path) as f:
        text = f.read()
    versions = {}
    stages = {}
    current = None
    for line in text.splitlines(keepends=True):
        header = re.match(r"^#\[(\w+)\]\s*$", line)
        if header:
            current = header.group(1)
            if current != "versions":
                stages[current] = ""
            continue
        if current == "versions":
            entry = re.match(r'^\s*(\w+)\s*=\s*"(.*)";', line)
            if entry:
                versions[entry.group(1)] = entry.group(2).replace("\\n", "\n")
        elif current is not None:
            stages[current] += line
    if not stages:  # No sections: a single compute shader.
        stages["compute"] = text
    return versions or {"": ""}, stages


def main():
    glslang = find_glslang()
    if not glslang:
        sys.exit("glslang not found; pass its path as the first argument")
    failures = 0
    checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        for shader in SHADERS:
            versions, stages = parse(os.path.join(MODULE_DIR, shader))
            for version, version_define in versions.items():
                for options in OPTION_SETS:
                    for stage, source in stages.items():
                        code = source.replace("#VERSION_DEFINES", version_define + "\n" + options)
                        code = inline_includes(code, MODULE_DIR)
                        out = os.path.join(tmp, "shader." + STAGES[stage])
                        with open(out, "w") as f:
                            f.write(code)
                        result = subprocess.run(
                            [glslang, "-V", "--target-env", "vulkan1.1", "-o", os.devnull, out],
                            capture_output=True,
                            text=True,
                        )
                        checked += 1
                        if result.returncode != 0:
                            failures += 1
                            option_names = " ".join(re.findall(r"#define (USE_\w+)", options)) or "no options"
                            print(f"FAIL {shader} [{version or 'default'}] {stage} ({option_names}):")
                            for line in (result.stdout + result.stderr).splitlines():
                                if "ERROR" in line or "error" in line:
                                    print("   ", line.replace(out, shader))
    print(f"{checked - failures}/{checked} shader variants compiled")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
