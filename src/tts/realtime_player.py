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
from src.core.SharedLock import SharedLock as lock 
from src.core.SharedAudio import AudioManager as Audio
from src.core.SharedAudio import get_audio_manager
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
        self._aborted = False
        
        # Find ffmpeg binary
        self.ffmpeg_path = self._find_ffmpeg()
        # if self.ffmpeg_path:
        #     print(f"Found ffmpeg in PATH: {self.ffmpeg_path}")
        # else:
        #     print("Warning: ffmpeg not found in PATH. Audio playback may not work.")
        self.listen_lock = lock()
        self.audio_manager = get_audio_manager()
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
                "E:\\ffmpeg\\bin\\ffmpeg.exe",
                "D:\\Apps\\ffmpeg\\bin\\ffmpeg.exe"
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
        self._aborted = False

    def start(self):
        """Start the player"""
        if not self.ffmpeg_path:
            print("Error: Cannot start player without ffmpeg")
            return False
        print("before pyaudio")  
        # with self.listen_lock:
        #     self._player = pyaudio.PyAudio()  # initialize pyaudio to play audio
        #with self.listen_lock:
        self._stream = self.audio_manager.get_output_stream(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=22050
                    )
        print("after pyaudio")
        # self._stream = self._player.open(
        #     format=pyaudio.paInt16, channels=1, rate=22050,
        #     output=True)  # initialize pyaudio stream
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
            self.stop_event.clear()
            self._aborted = False
            return True
        except subprocess.CalledProcessError as e:
            print(f'An error occurred: {e}')
            return False
        except FileNotFoundError:
            print(f"Error: Could not execute ffmpeg at {self.ffmpeg_path}")
            return False

    def stop(self):
        """Gracefully stop the player (may play out small buffered audio)."""
        try:
            self.stop_event.set()
            if self.ffmpeg_process and self.ffmpeg_process.stdin:
                try:
                    self.ffmpeg_process.stdin.close()
                except Exception:
                    pass
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.terminate()
                except Exception:
                    pass
            if self.play_thread:
                try:
                    self.play_thread.join(timeout=0.5)
                except Exception:
                    pass
            # Do not close shared stream here; leave it to AudioManager to reset on demand
            if self.verbose:
                print('MP3 audio player stopped')
        except Exception as e:
            # Capture any exceptions during cleanup
            print(f'An error occurred during player shutdown: {e}')

    def finalize(self, timeout: float = 3.0):
        """Finish playback cleanly by closing encoder input and waiting for decode/play to drain.

        This should be used on normal completion (non-interrupt) so that the last chunk of audio
        is not cut off. It closes ffmpeg stdin to signal EOF, waits for the playback thread
        to finish reading stdout and writing to the audio device, and then waits for ffmpeg
        to exit. It does not reset or close the shared output stream.
        """
        try:
            # Signal EOF to ffmpeg so it can flush and finish decoding
            if self.ffmpeg_process and self.ffmpeg_process.stdin:
                try:
                    self.ffmpeg_process.stdin.close()
                except Exception:
                    pass

            # Wait for play thread to drain stdout to the audio device
            if self.play_thread:
                try:
                    self.play_thread.join(timeout=timeout)
                except Exception:
                    pass

            # Give ffmpeg a moment to exit on its own; terminate if it lingers
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.wait(timeout=0.5)
                except Exception:
                    try:
                        self.ffmpeg_process.terminate()
                    except Exception:
                        pass
        except Exception as e:
            if self.verbose:
                print(f'Finalize error: {e}')

    def abort(self):
        """Immediately abort playback: drop buffers and kill decoding."""
        try:
            self._aborted = True
            self.stop_event.set()
            # Kill ffmpeg immediately
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.kill()
                except Exception:
                    pass
            # Attempt to stop playback thread quickly
            if self.play_thread:
                try:
                    self.play_thread.join(timeout=1.0)
                except Exception:
                    pass
            self.ffmpeg_process = None
        except Exception as e:
            print(f'Abort error: {e}')

    def play_audio(self):
        """Read and play audio data from ffmpeg's output"""
        # Play audio with PCM data decoded by ffmpeg
        try:
            while not self.stop_event.is_set():
                if self.ffmpeg_process is None:
                    break
                if self.ffmpeg_process.poll() is not None:
                    break
                pcm_data = self.ffmpeg_process.stdout.read(1024)
                if pcm_data:
                    try:
                        self._stream.write(pcm_data)
                    except Exception:
                        break
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
        if self._aborted:
            return
        if not self.ffmpeg_process:
            print("Error: ffmpeg process not started")
            return
        if self.stop_event.is_set():
            return
        # If ffmpeg has already exited, skip writes to avoid BrokenPipe
        try:
            if self.ffmpeg_process.poll() is not None:
                return
        except Exception:
            pass
            
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
            # ffmpeg is gone; ignore further writes
            pass
        except Exception as e:
            # Capture any exceptions during writing
            print(f'Error when writing audio data: {type(e).__name__}: {e}')


