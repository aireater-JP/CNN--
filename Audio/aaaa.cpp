#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include <iostream>
#include <vector>
#include <fftw3.h>
#include <cmath>
#include <complex>
#include <algorithm>

// WAVファイルを読み込む関数
int readWavFile(const char *filename, std::vector<double> &data, unsigned int &sampleRate, unsigned int &channels)
{
    drwav wav;
    if (!drwav_init_file(&wav, filename, nullptr))
    {
        std::cerr << "Failed to open WAV file: " << filename << std::endl;
        return -1;
    }

    sampleRate = wav.sampleRate;
    channels = wav.channels;
    size_t totalSampleCount = wav.totalPCMFrameCount * wav.channels;
    std::vector<float> floatData(totalSampleCount);

    drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, floatData.data());
    drwav_uninit(&wav);

    data.resize(totalSampleCount);
    for (size_t i = 0; i < totalSampleCount; ++i)
    {
        data[i] = floatData[i];
    }

    return 0;
}

// WAVファイルを書き込む関数
int writeWavFile(const char *filename, const std::vector<double> &data, unsigned int sampleRate, unsigned int channels)
{
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = channels;
    format.sampleRate = sampleRate;
    format.bitsPerSample = 16;

    drwav wav;
    if (!drwav_init_file_write(&wav, filename, &format, nullptr))
    {
        std::cerr << "Failed to open WAV file for writing: " << filename << std::endl;
        return -1;
    }

    // double配列をint16_tに変換して書き込みます
    std::vector<int16_t> int16_data(data.size());
    for (size_t i = 0; i < data.size(); ++i)
    {
        double sample = data[i] * 32767.0;

        // クリッピング
        if (sample > 32767.0)
        {
            sample = 32767.0;
        }
        else if (sample < -32768.0)
        {
            sample = -32768.0;
        }

        int16_data[i] = static_cast<int16_t>(sample);
    }

    drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, data.size() / channels, int16_data.data());
    if (framesWritten != data.size() / channels)
    {
        std::cerr << "Failed to write all samples to WAV file: " << filename << std::endl;
        drwav_uninit(&wav);
        return -1;
    }

    drwav_uninit(&wav);
    return 0;
}

// フーリエ変換を行いスペクトログラムを生成する関数
void computeSpectrogram(const std::vector<double> &input, std::vector<std::vector<std::complex<double>>> &spectrogram, size_t fft_size, size_t hop_size)
{
    size_t num_frames = (input.size() - fft_size) / hop_size + 1;
    spectrogram.resize(num_frames, std::vector<std::complex<double>>(fft_size / 2 + 1));

    fftw_plan plan = fftw_plan_dft_r2c_1d(fft_size, nullptr, nullptr, FFTW_ESTIMATE);
    std::vector<double> window(fft_size);
    for (size_t i = 0; i < fft_size; ++i)
        // window[i] = 0.5 * (1 - cos(2 * M_PI * i / (fft_size - 1))); // ハニング窓
        window[i] = 0.42 - 0.5 * cos(2 * M_PI * i / (fft_size - 1)) + 0.08 * cos(4 * M_PI * i / (fft_size - 1));

    std::vector<double> buffer(fft_size);
    std::vector<std::complex<double>> fft_result(fft_size / 2 + 1);
    for (size_t i = 0; i < num_frames; ++i)
    {
        for (size_t j = 0; j < fft_size; ++j)
            buffer[j] = input[i * hop_size + j] * window[j];

        fftw_execute_dft_r2c(plan, buffer.data(), reinterpret_cast<fftw_complex *>(fft_result.data()));
        spectrogram[i] = fft_result;
    }
    fftw_destroy_plan(plan);
}

// グリフィン・リムアルゴリズムを使用してスペクトログラムから音声信号を生成する関数
void griffinLim(const std::vector<std::vector<std::complex<double>>> &spectrogram, std::vector<double> &output, size_t fft_size, size_t hop_size, int iterations)
{
    size_t num_frames = spectrogram.size();
    size_t output_size = (num_frames - 1) * hop_size + fft_size;
    output.assign(output_size, 0);

    fftw_plan plan_forward = fftw_plan_dft_r2c_1d(fft_size, nullptr, nullptr, FFTW_ESTIMATE);
    fftw_plan plan_inverse = fftw_plan_dft_c2r_1d(fft_size, nullptr, nullptr, FFTW_ESTIMATE);

    std::vector<double> window(fft_size);
    for (size_t i = 0; i < fft_size; ++i)
        // window[i] = 0.5 * (1 - cos(2 * M_PI * i / (fft_size - 1))); // ハニング窓
        window[i] = 0.42 - 0.5 * cos(2 * M_PI * i / (fft_size - 1)) + 0.08 * cos(4 * M_PI * i / (fft_size - 1));

    std::vector<double> buffer(fft_size);
    std::vector<std::complex<double>> fft_input(fft_size / 2 + 1);
    std::vector<std::complex<double>> fft_output(fft_size / 2 + 1);

    for (int iter = 0; iter < iterations; ++iter)
    {
        for (size_t i = 0; i < num_frames; ++i)
        {
            for (size_t j = 0; j < fft_size; ++j)
                buffer[j] = output[i * hop_size + j] * window[j];

            fftw_execute_dft_r2c(plan_forward, buffer.data(), reinterpret_cast<fftw_complex *>(fft_input.data()));
            for (size_t k = 0; k < fft_size / 2 + 1; ++k)
                fft_input[k] = spectrogram[i][k] * std::polar(1.0, std::arg(fft_input[k]));

            fftw_execute_dft_c2r(plan_inverse, reinterpret_cast<fftw_complex *>(fft_input.data()), buffer.data());
            for (size_t j = 0; j < fft_size; ++j)
                output[i * hop_size + j] += buffer[j] * window[j] / fft_size;
        }
    }

    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_inverse);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_wav> <output_wav>" << std::endl;
        return -1;
    }

    const char *input_wav = argv[1];
    const char *output_wav = argv[2];
    unsigned int sampleRate, channels;
    std::vector<double> data;

    // WAVファイルの読み込み
    if (readWavFile(input_wav, data, sampleRate, channels) != 0)
        return -1;

    // パラメータ設定
    size_t fft_size = 1024 * 8;
    size_t hop_size = fft_size / 16;
    int iterations = 64;

    // スペクトログラムの計算
    std::vector<std::vector<std::complex<double>>> spectrogram;
    computeSpectrogram(data, spectrogram, fft_size, hop_size);

    // グリフィン・リムアルゴリズムで音声信号を生成
    std::vector<double> output;
    griffinLim(spectrogram, output, fft_size, hop_size, iterations);

    // WAVファイルの書き込み
    if (writeWavFile(output_wav, output, sampleRate, channels) != 0)
        return -1;

    std::cout << "Successfully processed " << input_wav << " and saved to " << output_wav << std::endl;
    return 0;
}
