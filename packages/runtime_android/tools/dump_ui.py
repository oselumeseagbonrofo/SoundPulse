import os
import subprocess
import time
import xml.etree.ElementTree as ET
import re

def run_adb(args):
    cmd = ["adb"] + args
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()

def main():
    print("Waking up device...")
    run_adb(["shell", "input", "keyevent", "KEYCODE_WAKE"])
    run_adb(["shell", "input", "keyevent", "KEYCODE_MENU"])
    
    # Do not launch app via monkey to avoid resetting tab index
    # print("Launching SoundPulse app...")
    # run_adb(["shell", "monkey", "-p", "com.oselumeseagbonrofo.soundpulse", "-c", "android.intent.category.LAUNCHER", "1"])
    # time.sleep(6)
    
    print("Dumping UI layout...")
    xml_path_device = "/data/local/tmp/window_dump.xml"
    run_adb(["shell", "uiautomator", "dump", xml_path_device])
    
    local_xml = "window_dump.xml"
    run_adb(["pull", xml_path_device, local_xml])
    print(f"Pulled UI layout to {local_xml}")
    
    if not os.path.exists(local_xml):
        print("Failed to pull UI dump!")
        return
        
    tree = ET.parse(local_xml)
    root = tree.getroot()
    
    print("\nClickable Elements Found:")
    print("-" * 80)
    for node in root.iter('node'):
        clickable = node.get('clickable') == 'true'
        text = node.get('text', '')
        resource_id = node.get('resource-id', '')
        content_desc = node.get('content-desc', '')
        bounds = node.get('bounds', '')
        
        if clickable or text or content_desc:
            print(f"Text: {text!r} | Desc: {content_desc!r} | ID: {resource_id!r} | Clickable: {clickable} | Bounds: {bounds}")
            
if __name__ == "__main__":
    main()
