#!/usr/bin/env python
"""
RealtimeMp3Player - A simple class for playing MP3 audio streams in real-time
Based on Alibaba Cloud's official example
"""

import subprocess
import threading
import pyaudio
import os
import shutil
import sys


class RealtimeMp3Player:
    """Player for streaming MP3 audio in real-time using ffmpeg and pyaudio"""
    
    def __init__(self, verbose=False):
        """Initialize the player
        
        Args:
            verbose: Whether to print debug messages
        """
        self.ffmpeg_process = None
        self._stream = None
        self._player = None
        self.play_thread = None
        self.stop_event = threading.Event()
        self.verbose = verbose
        
        # Find ffmpeg binary
        self.ffmpeg_path = self._find_ffmpeg()
        # if self.ffmpeg_path:
        #     print(f"Found ffmpeg in PATH: {self.ffmpeg_path}")
        # else:
        #     print("Warning: ffmpeg not found in PATH. Audio playback may not work.")

    def _find_ffmpeg(self):
        """Find the ffmpeg binary in the system PATH"""
        ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        
        # First check if ffmpeg is in the PATH
        ffmpeg_path = shutil.which(ffmpeg_name)
        if ffmpeg_path:
            return ffmpeg_path
            
        # On Windows, try some common installation locations
        if sys.platform == "win32":
            common_paths = [
                "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
                "C:\\ffmpeg\\bin\\ffmpeg.exe",
                "E:\\ffmpeg\\bin\\ffmpeg.exe"
            ]
            
            for path in common_paths:
                if os.path.isfile(path):
                    return path
        
        return None

    def reset(self):
        """Reset the player state"""
        self.ffmpeg_process = None
        self._stream = None
        self._player = None
        self.play_thread = None
        self.stop_event = threading.Event()

    def start(self):
        """Start the player"""
        if not self.ffmpeg_path:
            print("Error: Cannot start player without ffmpeg")
            return False
            
        self._player = pyaudio.PyAudio()  # initialize pyaudio to play audio
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=22050,
            output=True)  # initialize pyaudio stream
        try:
            self.ffmpeg_process = subprocess.Popen(
                [
                    self.ffmpeg_path, '-i', 'pipe:0', '-f', 's16le', '-ar', '22050',
                    '-ac', '1', 'pipe:1'
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )  # initialize ffmpeg to decode mp3
            if self.verbose:
                print('MP3 audio player started')
            return True
        except subprocess.CalledProcessError as e:
            print(f'An error occurred: {e}')
            return False
        except FileNotFoundError:
            print(f"Error: Could not execute ffmpeg at {self.ffmpeg_path}")
            return False

    def stop(self):
        """Stop the player"""
        try:
            if self.ffmpeg_process and self.ffmpeg_process.stdin:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait()
            if self.play_thread:
                self.play_thread.join()
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            if self._player:
                self._player.terminate()
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
            if self.verbose:
                print('MP3 audio player stopped')
        except Exception as e:
            # Capture any exceptions during cleanup
            print(f'An error occurred during player shutdown: {e}')

    def play_audio(self):
        """Read and play audio data from ffmpeg's output"""
        # Play audio with PCM data decoded by ffmpeg
        try:
            while not self.stop_event.is_set():
                pcm_data = self.ffmpeg_process.stdout.read(1024)
                if pcm_data:
                    self._stream.write(pcm_data)
                else:
                    break
        except Exception as e:
            # Capture any exceptions during playback
            print(f'An error occurred during playback: {e}')

    def write(self, data: bytes) -> None:
        """Write audio data to the player
        
        Args:
            data: MP3 audio data to be played
        """
        if not self.ffmpeg_process:
            print("Error: ffmpeg process not started")
            return
            
        try:
            self.ffmpeg_process.stdin.write(data)
            self.ffmpeg_process.stdin.flush()  # Ensure data is sent to ffmpeg
            
            if self.play_thread is None:
                # Initialize play thread
                self._stream.start_stream()
                self.play_thread = threading.Thread(target=self.play_audio)
                self.play_thread.daemon = True
                self.play_thread.start()
        except BrokenPipeError:
            print("Error: Broken pipe when writing to ffmpeg")
        except Exception as e:
            # Capture any exceptions during writing
            print(f'Error when writing audio data: {type(e).__name__}: {e}')