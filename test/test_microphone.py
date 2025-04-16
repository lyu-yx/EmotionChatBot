#!/usr/bin/env python
"""
Test script for microphone functionality.
This script checks if the microphone is properly detected and working.
"""

import os
import sys
import time
import pyaudio
import wave
import speech_recognition as sr

def list_available_microphones():
    """List all available microphones in the system"""
    print("=== Available Microphones ===")
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    
    if num_devices <= 0:
        print("No microphones detected!")
        return False
    
    found_mics = False
    
    for i in range(0, num_devices):
        device = p.get_device_info_by_host_api_device_index(0, i)
        if device.get('maxInputChannels') > 0:
            print(f"Mic ID {i}: {device.get('name')}")
            found_mics = True
    
    p.terminate()
    
    if not found_mics:
        print("No input devices found!")
        return False
    
    return True

def test_microphone_recording():
    """Test recording from the default microphone using PyAudio"""
    print("\n=== Testing Microphone Recording ===")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Set parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 5
    
    # Create a temporary file to store the recording
    temp_file = "test_recording.wav"
    
    print(f"Recording for {RECORD_SECONDS} seconds...")
    
    try:
        # Open stream for recording
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        frames = []
        
        # Record for RECORD_SECONDS
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            # Print progress
            if i % 10 == 0:
                print(".", end="", flush=True)
        
        print("\nFinished recording.")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Save the recorded audio to a WAV file
        wf = wave.open(temp_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Recording saved to {temp_file}")
        print("Recording test successful")
        
        # Test audio levels to make sure recording isn't just silence
        check_audio_levels(frames)
        
        p.terminate()
        return True
        
    except Exception as e:
        print(f"\nError during recording: {e}")
        p.terminate()
        return False

def check_audio_levels(frames):
    """Check if the recorded audio has any significant sound levels"""
    # Convert frames to integers and calculate RMS
    total_samples = 0
    sum_squares = 0
    
    for frame in frames:
        # Process each 2-byte sample in the frame
        for i in range(0, len(frame), 2):
            try:
                sample = int.from_bytes(frame[i:i+2], byteorder='little', signed=True)
                sum_squares += sample * sample
                total_samples += 1
            except:
                continue
    
    if total_samples == 0:
        print("WARNING: No audio samples found!")
        return False
    
    rms = (sum_squares / total_samples) ** 0.5
    print(f"\nAverage audio level (RMS): {rms:.2f}")
    
    if rms < 100:
        print("WARNING: Audio levels very low. Microphone may not be picking up sound properly.")
        return False
    else:
        print("Audio levels seem OK.")
        return True

def test_speech_recognizer():
    """Test if speech_recognition library can access the microphone"""
    print("\n=== Testing Speech Recognition Microphone Access ===")
    
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speech recognition can access microphone")
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Microphone input level (energy threshold):", r.energy_threshold)
            
            if r.energy_threshold < 100:
                print("WARNING: Energy threshold very low, check microphone levels")
                return False
            return True
    except Exception as e:
        print(f"Error accessing microphone with speech_recognition: {e}")
        return False

if __name__ == "__main__":
    print("=== Microphone Test Utility ===")
    
    print("\nStep 1: Checking available microphones...")
    mics_available = list_available_microphones()
    
    if not mics_available:
        print("\nERROR: No microphones detected in the system.")
        print("Please check your microphone connection and drivers.")
        sys.exit(1)
    
    print("\nStep 2: Testing speech_recognition microphone access...")
    sr_test = test_speech_recognizer()
    
    print("\nStep 3: Testing microphone recording...")
    recording_test = test_microphone_recording()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Microphones detected: {'YES' if mics_available else 'NO'}")
    print(f"Speech recognition access: {'SUCCESS' if sr_test else 'FAILED'}")
    print(f"Recording test: {'SUCCESS' if recording_test else 'FAILED'}")
    
    if not sr_test or not recording_test:
        print("\nTroubleshooting steps:")
        print("1. Ensure your microphone is properly connected")
        print("2. Check that your microphone isn't muted in Windows sound settings")
        print("3. Check microphone privacy settings in Windows (Settings > Privacy > Microphone)")
        print("4. Try selecting a different microphone as default in Windows sound settings")
        print("5. Increase microphone volume in sound control panel")
    else:
        print("\nAll tests passed! Your microphone appears to be working correctly.")