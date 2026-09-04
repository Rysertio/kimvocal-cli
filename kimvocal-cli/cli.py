import argparse
import os
import sys
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort

def separate_vocals(model_path: str, input_wav: str, output_dir: str, device: str):
    TARGET_SR = 44100
    N_FFT = 6144
    HOP_LENGTH = 1024
    DIM_F = 3072  # Frequency bins (N_FFT // 2)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_wav))[0]
    output_vocal = os.path.join(output_dir, f"{base_name}_vocals.wav")
    output_inst = os.path.join(output_dir, f"{base_name}_instrumental.wav")

    print(f"Loading '{input_wav}'...")
    audio, sr = librosa.load(input_wav, sr=TARGET_SR, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])

    print("Computing STFT...")
    stft_L = librosa.stft(audio[0], n_fft=N_FFT, hop_length=HOP_LENGTH)
    stft_R = librosa.stft(audio[1], n_fft=N_FFT, hop_length=HOP_LENGTH)

    mag_L, phase_L = np.abs(stft_L[:DIM_F, :]), np.angle(stft_L[:DIM_F, :])
    mag_R, phase_R = np.abs(stft_R[:DIM_F, :]), np.angle(stft_R[:DIM_F, :])

    mix_mag = np.stack([mag_L, mag_R], axis=0)[np.newaxis, ...].astype(np.float32)

    # Configure hardware execution providers
    providers = ['CPUExecutionProvider']
    if device == 'openvino':
        providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
    elif device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    print(f"Running ONNX Inference via {providers[0]}...")
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    vocal_mag_mask = session.run([output_name], {input_name: mix_mag})[0][0]

    print("Reconstructing audio waveforms...")
    vocal_stft_L = vocal_mag_mask[0] * np.exp(1j * phase_L)
    vocal_stft_R = vocal_mag_mask[1] * np.exp(1j * phase_R)

    vocal_time_L = librosa.istft(vocal_stft_L, hop_length=HOP_LENGTH, length=audio.shape[1])
    vocal_time_R = librosa.istft(vocal_stft_R, hop_length=HOP_LENGTH, length=audio.shape[1])
    vocal_stereo = np.stack([vocal_time_L, vocal_time_R], axis=-1)

    inst_stereo = audio.T - vocal_stereo

    sf.write(output_vocal, vocal_stereo, TARGET_SR)
    sf.write(output_inst, inst_stereo, TARGET_SR)
    print(f"Done!\n- Vocals: {output_vocal}\n- Instrumental: {output_inst}")

def main():
    parser = argparse.ArgumentParser(
        description="Separate vocals and instrumentals using Kim_Vocal_2.onnx (PyTorch-free)."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input WAV file")
    parser.add_argument("-m", "--model", default="Kim_Vocal_2.onnx", help="Path to Kim_Vocal_2.onnx model")
    parser.add_argument("-o", "--output-dir", default="./output", help="Directory to save output files")
    parser.add_argument("-d", "--device", choices=["cpu", "openvino", "cuda"], default="cpu", help="Execution provider")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)

    separate_vocals(
        model_path=args.model,
        input_wav=args.input,
        output_dir=args.output_dir,
        device=args.device
    )

if __name__ == "__main__":
    main()
