#!/usr/bin/env python
"""
Test script for Dashscope ASR (Automatic Speech Recognition)
-----------------------------------------------------------
This script tests the Dashscope speech recognition functionality.
"""

import os
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.asr.speech_recognition_engine import DashscopeSpeechRecognizer

def test_dashscope_asr():
    """Test Dashscope Speech Recognition with both languages"""
    print("\n=== Testing Dashscope Speech Recognition ===")
    
    # Test Chinese recognition
    test_dashscope_language("zh-cn")
    
    # Wait a bit before next test
    # time.sleep(2)
    
    # Test English recognition
    # test_dashscope_language("en-us")

def test_dashscope_language(language):
    """Test Dashscope Speech Recognition with specific language"""
    lang_name = "Chinese" if language.startswith("zh") else "English"
    print(f"\n--- Testing {lang_name} Recognition with Dashscope ---")
    
    try:
        # Initialize the recognizer with the specified language
        print(f"Initializing DashscopeSpeechRecognizer for {lang_name}...")
        start_time = time.time()
        recognizer = DashscopeSpeechRecognizer(language=language)
        end_time = time.time()
        print(f"Initialization time: {end_time - start_time:.2f} seconds")
        
        print(f"Please speak in {lang_name} after the prompt...")
        time.sleep(1)
        
        # Test recognition
        print(f"Recording and recognizing speech...")
        start_time = time.time()
        result = recognizer.recognize_from_microphone()
        end_time = time.time()
        
        if result["success"]:
            print(f"\nRecognition result ({lang_name}): {result['text']}")
            print(f"Total processing time: {end_time - start_time:.2f} seconds")
            print(f"Recognition engine: {result['engine']}")
            
            # Check if the result is from simulation fallback
            if result["engine"] == "simulation_fallback":
                print(f"NOTE: This was a simulated response because the model processing failed.")
        else:
            print(f"\nRecognition failed ({lang_name}): {result['error']}")
        
        return result["success"]
    
    except Exception as e:
        print(f"\nError during {lang_name} recognition: {e}")
        return False

def check_microphone():
    """Check if microphone is working properly"""
    print("\n=== Checking Microphone ===")
    
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        
        # List available audio devices
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        print(f"Found {numdevices} audio devices")
        
        # Check if there's at least one input device
        input_devices = []
        for i in range(0, numdevices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                input_devices.append((i, device_info.get('name')))
        
        if input_devices:
            print("✓ Found input devices:")
            for idx, name in input_devices:
                print(f"  #{idx}: {name}")
                
            try:
                default = p.get_default_input_device_info()
                print(f"\n✓ Default input device: #{default['index']}: {default['name']}")
            except:
                print("✗ No default input device set")
                
            # Try to open the default input device
            try:
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024,
                    start=False
                )
                print("✓ Successfully opened audio stream")
                stream.close()
            except Exception as e:
                print(f"✗ Failed to open audio stream: {e}")
        else:
            print("✗ No input devices found")
        
        p.terminate()
        
    except Exception as e:
        print(f"✗ Error checking microphone: {e}")

if __name__ == "__main__":
    print("=== ASR Test ===")
    print("This script will test the Dashscope speech recognition functionality.")
    
    # Check microphone first
    check_microphone()
    
    print("\nTesting Dashscope ASR...")
    print("The model will connect to Dashscope API for recognition.")
    test_dashscope_asr()
    
    print("\n=== Test Complete ===")
    print("\nNotes:")
    print("1. Make sure your DASHSCOPE_API_KEY is set in environment variables or config.json")
    print("2. If ASR fails, check for mic or connectivity issues")
    print("3. Ensure your microphone is working correctly")