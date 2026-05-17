import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SRC = ROOT / "packages" / "runtime_android" / "src" / "main" / "kotlin" / "com" / "soundpulse" / "runtime"
HARNESS_SRC = ROOT / "packages" / "runtime_android" / "src" / "test" / "kotlin" / "com" / "soundpulse" / "runtime" / "RuntimeVerificationHarness.kt"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def resolve_kotlin_bin(binary_name: str) -> str:
    found = shutil.which(binary_name)
    if found:
        return found

    if os.name == "nt":
        mapped_name = f"{binary_name}.bat"
        candidates = [
            Path(r"C:\Program Files\Android\Android Studio\plugins\Kotlin\kotlinc\bin") / mapped_name,
            Path(r"C:\Program Files (x86)\Android\Android Studio\plugins\Kotlin\kotlinc\bin") / mapped_name,
        ]
    else:
        candidates = [Path("/opt/android-studio/plugins/Kotlin/kotlinc/bin") / binary_name]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    fail(f"Unable to locate {binary_name}. Add it to PATH or install Kotlin compiler.")


def resolve_android_jar() -> str:
    sdk_root = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if not sdk_root:
        fail("ANDROID_HOME/ANDROID_SDK_ROOT is not set.")

    platforms_dir = Path(sdk_root) / "platforms"
    if not platforms_dir.exists():
        fail(f"Android SDK platforms directory not found: {platforms_dir}")

    candidates = sorted(platforms_dir.glob("android-*/android.jar"), key=lambda path: path.parent.name)
    if not candidates:
        fail(f"No android.jar found under {platforms_dir}")
    return str(candidates[-1])


def resolve_java_home() -> str:
    java_home = os.environ.get("JAVA_HOME")
    if java_home and (Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")).exists():
        return java_home

    if os.name == "nt":
        candidates = [
            Path(r"C:\Program Files\Android\Android Studio\jbr"),
            Path(r"C:\Program Files\Java\jdk-21"),
            Path(r"C:\Program Files\Java\jdk-17"),
        ]
    else:
        candidates = [Path("/usr/lib/jvm/default-java"), Path("/usr/lib/jvm/java-17-openjdk")]

    for candidate in candidates:
        java_exe = candidate / "bin" / ("java.exe" if os.name == "nt" else "java")
        if java_exe.exists():
            return str(candidate)

    fail("Unable to locate Java runtime for Kotlin compiler execution.")


def resolve_java_executable(java_home: str) -> str:
    java_name = "java.exe" if os.name == "nt" else "java"
    java_exe = Path(java_home) / "bin" / java_name
    if not java_exe.exists():
        fail(f"Java executable not found: {java_exe}")
    return str(java_exe)


def run_command(args: list[str], label: str, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="")
        fail(f"{label} failed")
    if completed.stdout:
        print(completed.stdout, end="")


def main() -> None:
    checks = [
        [shutil.which("python") or "python", "packages/runtime_android/tools/check_preprocessing_parity_fixture.py"],
        [shutil.which("python") or "python", "packages/runtime_android/tools/validate_inference_result.py"],
        [shutil.which("python") or "python", "packages/runtime_android/tools/validate_policy_invariants.py"],
    ]
    for command in checks:
        run_command(command, "Python contract check")

    kotlinc = resolve_kotlin_bin("kotlinc")
    android_jar = resolve_android_jar()
    java_home = resolve_java_home()
    java_exe = resolve_java_executable(java_home)
    kotlin_env = {
        "JAVA_HOME": java_home,
        "PATH": f"{Path(java_home) / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    }

    source_files = sorted(str(path) for path in RUNTIME_SRC.glob("*.kt"))
    if not source_files:
        fail(f"No runtime sources found under {RUNTIME_SRC}")
    source_files.append(str(HARNESS_SRC))
    kotlin_lib_dir = Path(kotlinc).resolve().parents[1] / "lib"
    kotlin_runtime_jars = [
        kotlin_lib_dir / "kotlin-stdlib.jar",
        kotlin_lib_dir / "kotlin-stdlib-jdk8.jar",
        kotlin_lib_dir / "kotlin-stdlib-jdk7.jar",
    ]
    missing_runtime = [str(path) for path in kotlin_runtime_jars if not path.exists()]
    if missing_runtime:
        fail(f"Kotlin runtime jars not found: {', '.join(missing_runtime)}")

    with tempfile.TemporaryDirectory(prefix="soundpulse-runtime-test-") as temp_dir:
        compile_cmd = [kotlinc, *source_files, "-cp", android_jar, "-d", temp_dir]
        run_command(compile_cmd, "Kotlin harness compilation", extra_env=kotlin_env)

        classpath_entries = [temp_dir, android_jar, *[str(path) for path in kotlin_runtime_jars]]
        classpath = os.pathsep.join(classpath_entries)
        run_cmd = [java_exe, "-cp", classpath, "com.soundpulse.runtime.RuntimeVerificationHarnessKt"]
        run_command(run_cmd, "Kotlin runtime harness", extra_env=kotlin_env)

    print("PASS: runtime verification checks succeeded")


if __name__ == "__main__":
    main()
