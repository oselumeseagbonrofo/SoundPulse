import os
import subprocess
import time
import xml.etree.ElementTree as ET
import re

def run_adb(args):
    cmd = ["adb"] + args
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"ADB Command failed: {' '.join(cmd)}")
        print(f"Error: {result.stderr.strip()}")
    return result.stdout.strip()

def dump_ui(local_xml="temp_dump.xml"):
    xml_path_device = "/data/local/tmp/window_dump.xml"
    run_adb(["shell", "uiautomator", "dump", xml_path_device])
    run_adb(["pull", xml_path_device, local_xml])
    if not os.path.exists(local_xml):
        return None
    return local_xml

def get_node_center(local_xml, text_pattern):
    try:
        tree = ET.parse(local_xml)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None
        
    for node in root.iter('node'):
        text = node.get('text', '')
        content_desc = node.get('content-desc', '')
        if re.search(text_pattern, text, re.IGNORECASE) or re.search(text_pattern, content_desc, re.IGNORECASE):
            bounds_str = node.get('bounds')
            match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                return (x1 + x2) // 2, (y1 + y2) // 2
    return None

def tap(x, y):
    print(f"Tapping at ({x}, {y})")
    run_adb(["shell", "input", "tap", str(x), str(y)])

def select_tab(tab_name, tab_x, tab_y):
    print(f"Selecting tab {tab_name}...")
    for attempt in range(4):
        tap(tab_x, tab_y)
        time.sleep(2.5)
        xml_file = dump_ui()
        if xml_file:
            if tab_name == "Live Feed":
                if get_node_center(xml_file, "Tiny CNN") or get_node_center(xml_file, "YAMNet"):
                    return xml_file
                else:
                    # Swipe down to scroll up to the top of the tab
                    print("Live Feed settings not found. Scrolling up...")
                    run_adb(["shell", "input", "swipe", "500", "500", "500", "1800", "300"])
                    time.sleep(1.5)
                    xml_file = dump_ui()
                    if xml_file and (get_node_center(xml_file, "Tiny CNN") or get_node_center(xml_file, "YAMNet")):
                        return xml_file
            elif tab_name == "Settings" and (get_node_center(xml_file, "Research Evaluation & Benchmarking") or get_node_center(xml_file, "Sound Detection Preferences")):
                return xml_file
        print(f"Retrying tab selection (attempt {attempt+1}/4)...")
        time.sleep(1.5)
    return None

def find_and_tap_benchmark(xml_file):
    # Try finding button directly first
    benchmark_btn = get_node_center(xml_file, "Run On-Device Model Benchmark")
    if benchmark_btn:
        print("Found benchmark button directly.")
        tap(benchmark_btn[0], benchmark_btn[1])
        return True
    
    # Try swiping/scrolling down
    print("Button not found. Scrolling down settings page...")
    run_adb(["shell", "input", "swipe", "500", "1800", "500", "800", "300"])
    time.sleep(1.5)
    xml_file = dump_ui()
    if xml_file:
        benchmark_btn = get_node_center(xml_file, "Run On-Device Model Benchmark")
        if benchmark_btn:
            print("Found benchmark button after scrolling.")
            tap(benchmark_btn[0], benchmark_btn[1])
            return True
            
    print("Failed to find benchmark button even after scrolling.")
    return False

def main():
    print("=== STARTING AUTOMATED RUNNER ===")
    
    # Clear logcat first
    run_adb(["logcat", "-c"])
    
    # 1. Wake up and launch app
    run_adb(["shell", "input", "keyevent", "KEYCODE_WAKE"])
    run_adb(["shell", "input", "keyevent", "KEYCODE_MENU"])
    print("Launching app...")
    run_adb(["shell", "monkey", "-p", "com.oselumeseagbonrofo.soundpulse", "-c", "android.intent.category.LAUNCHER", "1"])
    
    # Wait for app to load
    time.sleep(10)
    
    # Poll for screen to load (up to 4 retries)
    xml_file = None
    for attempt in range(4):
        xml_file = dump_ui()
        if xml_file:
            # Check if any recognizable button is on screen
            if get_node_center(xml_file, "Live Feed") or get_node_center(xml_file, "^OFF$") or get_node_center(xml_file, "^ON$"):
                print("App UI loaded successfully.")
                break
        print(f"Waiting for UI to load (attempt {attempt+1}/4)...")
        time.sleep(4)

    if not xml_file:
        print("Failed to dump UI. Exiting.")
        return
        
    # Check if OFF is showing. If yes, tap to turn ON monitoring
    off_center = get_node_center(xml_file, "^OFF$")
    if off_center:
        print("Monitoring is OFF. Toggling ON...")
        tap(off_center[0], off_center[1])
        time.sleep(4)
        xml_file = dump_ui()
        
    # === BENCHMARK TINY CNN ===
    print("\n--- Setting up Tiny CNN benchmark ---")
    xml_file = select_tab("Live Feed", 180, 2160)
    if not xml_file:
        print("Error: Could not switch to Live Feed tab. Exiting.")
        return
        
    # Select Tiny CNN chip
    tiny_cnn_chip = get_node_center(xml_file, "Tiny CNN")
    if tiny_cnn_chip:
        print("Selecting Tiny CNN Backbone...")
        tap(tiny_cnn_chip[0], tiny_cnn_chip[1])
        time.sleep(2)
        
    # Switch to Settings tab via top right button
    xml_file = select_tab("Settings", 979, 155)
    if not xml_file:
        print("Error: Could not switch to Settings tab. Exiting.")
        return
    
    # Run Tiny CNN Benchmark
    print("Running Tiny CNN Benchmark...")
    if find_and_tap_benchmark(xml_file):
        print("Waiting 20 seconds for benchmark to complete...")
        time.sleep(20)
        xml_file = dump_ui()
    else:
        print("Skipping Tiny CNN benchmark due to missing button.")
        
    # === BENCHMARK YAMNET ===
    print("\n--- Setting up YAMNet benchmark ---")
    # Tap Live Feed Tab
    xml_file = select_tab("Live Feed", 180, 2160)
    if not xml_file:
        print("Error: Could not switch to Live Feed tab. Exiting.")
        return
        
    # Select YAMNet chip
    yamnet_chip = get_node_center(xml_file, "YAMNet")
    if yamnet_chip:
        print("Selecting YAMNet Backbone...")
        tap(yamnet_chip[0], yamnet_chip[1])
        time.sleep(2)
    else:
        print("Warning: YAMNet chip not found on screen!")
        
    # Tap Settings button
    xml_file = select_tab("Settings", 979, 155)
    if not xml_file:
        print("Error: Could not switch to Settings tab. Exiting.")
        return
    
    # Run YAMNet Benchmark
    print("Running YAMNet Benchmark...")
    if find_and_tap_benchmark(xml_file):
        print("Waiting 20 seconds for benchmark to complete...")
        time.sleep(20)
        xml_file = dump_ui()
    else:
        print("Skipping YAMNet benchmark due to missing button.")
        
    # === RUN BATTERY DRAIN TEST ===
    print("\n--- Running brief Battery Drain Test ---")
    battery_start_btn = get_node_center(xml_file, "^Start$")
    if battery_start_btn:
        print("Starting continuous battery drain test...")
        tap(battery_start_btn[0], battery_start_btn[1])
        time.sleep(5)
        xml_file = dump_ui()
        
        # Stop battery drain test
        battery_stop_btn = get_node_center(xml_file, "^Stop$")
        if battery_stop_btn:
            print("Stopping battery drain test...")
            tap(battery_stop_btn[0], battery_stop_btn[1])
            time.sleep(2)
            
    # === PULL PREDICTIONS JSON ===
    print("\n--- Pulling benchmark results from device ---")
    ml_eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "evaluation"))
    
    tiny_device_path = "/sdcard/Android/data/com.oselumeseagbonrofo.soundpulse/files/Documents/tiny_cnn_benchmark_predictions.json"
    yamnet_device_path = "/sdcard/Android/data/com.oselumeseagbonrofo.soundpulse/files/Documents/yamnet_benchmark_predictions.json"
    
    local_tiny = os.path.join(ml_eval_dir, "tiny_cnn_benchmark_predictions.json")
    local_yamnet = os.path.join(ml_eval_dir, "yamnet_benchmark_predictions.json")
    
    print(f"Pulling Tiny CNN predictions to: {local_tiny}")
    run_adb(["pull", tiny_device_path, local_tiny])
    
    print(f"Pulling YAMNet predictions to: {local_yamnet}")
    run_adb(["pull", yamnet_device_path, local_yamnet])
    
    # Clean up temp dump
    if os.path.exists("temp_dump.xml"):
        os.remove("temp_dump.xml")
        
    print("\n=== AUTOMATED RUNNER FINISHED ===")

if __name__ == "__main__":
    main()
