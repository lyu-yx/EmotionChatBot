import pyaudio
# This class is to manage the pyaudio stream in different module, quote the same object to handle i/o to avoid potential conflict
class AudioManager:
    def __init__(self):
        self._pyaudio = pyaudio.PyAudio()
        self._mic_stream = None
        self._output_stream = None

    def get_mic_stream(self, format=pyaudio.paInt16, channels=1, rate=16000, block_size=1024):
        if self._mic_stream is None:
            self._mic_stream = self._pyaudio.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=block_size
            )
        return self._mic_stream

    def get_output_stream(self, format=pyaudio.paInt16, channels=1, rate=22050):
        if self._output_stream is None:
            self._output_stream = self._pyaudio.open(
                format=format,
                channels=channels,
                rate=rate,
                output=True
            )
        return self._output_stream

    def terminate(self):
        if self._mic_stream:
            self._mic_stream.stop_stream()
            self._mic_stream.close()
            self._mic_stream = None
        if self._output_stream:
            self._output_stream.stop_stream()
            self._output_stream.close()
            self._output_stream = None
        self._pyaudio.terminate()

# 单例入口
_audio_manager_instance = None

def get_audio_manager():
    global _audio_manager_instance
    if _audio_manager_instance is None:
        _audio_manager_instance = AudioManager()
    return _audio_manager_instance