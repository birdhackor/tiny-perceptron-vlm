sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends ffmpeg
python - <<'PY'
import io
import subprocess
import wave

audio = subprocess.run(
    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
     "flite=text=ready:voice=slt", "-t", "0.25", "-ac", "1", "-ar", "16000",
     "-acodec", "pcm_s16le", "-f", "wav", "pipe:1"],
    check=True, capture_output=True, timeout=30,
).stdout
with wave.open(io.BytesIO(audio), "rb") as source:
    assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
    samples = source.readframes(16000)
    assert 1600 <= len(samples) <= 32000 and any(samples)
print("FFmpeg flite/slt produced nonempty mono PCM16 speech at 16 kHz before reservation")
PY
