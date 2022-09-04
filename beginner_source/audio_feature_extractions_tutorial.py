# -*- coding: utf-8 -*-
"""
오디오 특징 추출
=========================

``torchaudio``은 오디오 도메인에서 흔히 사용되는 특징 추출을 구현합니다. 
``torchaudio.functional``와 ``torchaudio.transforms``에서 사용 가능합니다.

``functional``은 독립형 함수들과 같은 특징들을 구현합니다. 
이것들은 스테이트리스(무상태)입니다.

``transforms`` 는 ``functional``과 ``torch.nn.Module``의 구현들을 이용하여 오브젝트와 같은 특징들을 구현합니다.
이것들은 TorchScript를 사용하여 직렬화할 수 있습니다.
"""

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
# 준비
# -----------
#
# .. note::
#
#    구글 코랩에서 이 튜토리얼을 실행할 때 필요한 패키지들을 설치해주세요.
#
#    .. code::
#
#       !pip install librosa
#
from IPython.display import Audio
import librosa
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

torch.random.manual_seed(0)

SAMPLE_SPEECH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")


def plot_waveform(waveform, sr, title="Waveform"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block=False)


######################################################################
# 오디오 특징 개요
# --------------------------
#
# 아래의 도표는 일반적인 오디오 특징들과 그것들을 생성하기 위한 torchaudio API 간의 관계를 보여줍니다.
#
# .. image:: https://download.pytorch.org/torchaudio/tutorial-assets/torchaudio_feature_extractions.png
#
# 사용 가능한 특징들의 전체 리스트를 보기 위해서는 다음 문서를 참조해주세요.
#


######################################################################
# 스펙트로그램
# -----------
#
# 시간에 따라 변하는 오디오 신호의 주파수 구성을 얻기 위해
# :py:func:`torchaudio.transforms.Spectrogram`을 사용할 수 있습니다.
#

SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)

plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="Original waveform")
Audio(SPEECH_WAVEFORM.numpy(), rate=SAMPLE_RATE)


######################################################################
#

n_fft = 1024
win_length = None
hop_length = 512

# 변환 정의
spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
)

######################################################################
#

# 변환 수행
spec = spectrogram(SPEECH_WAVEFORM)

######################################################################
#

plot_spectrogram(spec[0], title="torchaudio")

######################################################################
# GriffinLim
# ----------
#
# 스펙트로그램으로부터 파형을 복구하기 위해서 ``GriffinLim``을 사용할 수 있습니다.
#

torch.random.manual_seed(0)

n_fft = 1024
win_length = None
hop_length = 512

spec = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
)(SPEECH_WAVEFORM)

######################################################################
#

griffin_lim = T.GriffinLim(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
)

######################################################################
#

reconstructed_waveform = griffin_lim(spec)

######################################################################
#

plot_waveform(reconstructed_waveform, SAMPLE_RATE, title="Reconstructed")
Audio(reconstructed_waveform, rate=SAMPLE_RATE)

######################################################################
# 멜 필터 뱅크
# ---------------
#
# :py:func:`torchaudio.functional.melscale_fbanks`은 주파수 빈을 멜 스케일 빈으로
# 변환하기 위해 필터 뱅크를 생성합니다. 
#
# 이 함수는 입력 오디오/특징을 필요로 하지 않기 때문에, :py:func:`torchaudio.transforms`에 
# 동일한 변환은 없습니다.
#

n_fft = 256
n_mels = 64
sample_rate = 6000

mel_filters = F.melscale_fbanks(
    int(n_fft // 2 + 1),
    n_mels=n_mels,
    f_min=0.0,
    f_max=sample_rate / 2.0,
    sample_rate=sample_rate,
    norm="slaney",
)

######################################################################
#

plot_fbank(mel_filters, "Mel Filter Bank - torchaudio")

######################################################################
# librosa와의 비교
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 참고로 다음은 ``librosa``를 사용하여 동일하게 멜 필터를 얻는 방법입니다.
#

mel_filters_librosa = librosa.filters.mel(
    sr=sample_rate,
    n_fft=n_fft,
    n_mels=n_mels,
    fmin=0.0,
    fmax=sample_rate / 2.0,
    norm="slaney",
    htk=True,
).T

######################################################################
#

plot_fbank(mel_filters_librosa, "Mel Filter Bank - librosa")

mse = torch.square(mel_filters - mel_filters_librosa).mean().item()
print("Mean Square Difference: ", mse)

######################################################################
# 멜스펙트로그램
# --------------
#
# 멜 스케일의 스펙트로그램을 생성하는 것은 스펙트로그램을 만들고 멜 스케일 변환을
# 수행하는 것을 포함합니다. ``torchaudio`` 내의,
# :py:func:`torchaudio.transforms.MelSpectrogram` 함수가 이러한 기능을 제공합니다.
#

n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    onesided=True,
    n_mels=n_mels,
    mel_scale="htk",
)

melspec = mel_spectrogram(SPEECH_WAVEFORM)

######################################################################
#

plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")

######################################################################
# librosa와의 비교
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# 참고를 위해 ``librosa``를 사용하여 동일하게 멜 스케일 스펙트로그램을 생성하는 방법을
# 소개합니다.
#

melspec_librosa = librosa.feature.melspectrogram(
    y=SPEECH_WAVEFORM.numpy()[0],
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    n_mels=n_mels,
    norm="slaney",
    htk=True,
)

######################################################################
#

plot_spectrogram(melspec_librosa, title="MelSpectrogram - librosa", ylabel="mel freq")

mse = torch.square(melspec - melspec_librosa).mean().item()
print("Mean Square Difference: ", mse)

######################################################################
# MFCC
# ----
#

n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
    },
)

mfcc = mfcc_transform(SPEECH_WAVEFORM)

######################################################################
#

plot_spectrogram(mfcc[0])

######################################################################
# librosa와의 비교
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#

melspec = librosa.feature.melspectrogram(
    y=SPEECH_WAVEFORM.numpy()[0],
    sr=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    n_mels=n_mels,
    htk=True,
    norm=None,
)

mfcc_librosa = librosa.feature.mfcc(
    S=librosa.core.spectrum.power_to_db(melspec),
    n_mfcc=n_mfcc,
    dct_type=2,
    norm="ortho",
)

######################################################################
#

plot_spectrogram(mfcc_librosa)

mse = torch.square(mfcc - mfcc_librosa).mean().item()
print("Mean Square Difference: ", mse)

######################################################################
# LFCC
# ----
#

n_fft = 2048
win_length = None
hop_length = 512
n_lfcc = 256

lfcc_transform = T.LFCC(
    sample_rate=sample_rate,
    n_lfcc=n_lfcc,
    speckwargs={
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
    },
)

lfcc = lfcc_transform(SPEECH_WAVEFORM)
plot_spectrogram(lfcc[0])

######################################################################
# 피치(Pitch)
# -----
#

pitch = F.detect_pitch_frequency(SPEECH_WAVEFORM, SAMPLE_RATE)

######################################################################
#

def plot_pitch(waveform, sr, pitch):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sr
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

    axis2.legend(loc=0)
    plt.show(block=False)


plot_pitch(SPEECH_WAVEFORM, SAMPLE_RATE, pitch)

######################################################################
# 칼디 피치(Kaldi Pitch) (베타)
# ------------------
#
# 칼디 피치(Kaldi Pitch)는 자동 음성 인식(ASR) 애플리케이션을 위해 조정된 피치 감지 메커니즘입니다.
# 이것은 ``torchaudio``의 베타 기능이며 :py:func:`torchaudio.functional.compute_kaldi_pitch`로 사용할 수 있습니다.
#
# 1. A pitch extraction algorithm tuned for automatic speech recognition
#
#    Ghahremani, B. BabaAli, D. Povey, K. Riedhammer, J. Trmal and S.
#    Khudanpur
#
#    2014 IEEE International Conference on Acoustics, Speech and Signal
#    Processing (ICASSP), Florence, 2014, pp. 2494-2498, doi:
#    10.1109/ICASSP.2014.6854049.
#    [`abstract <https://ieeexplore.ieee.org/document/6854049>`__],
#    [`paper <https://danielpovey.com/files/2014_icassp_pitch.pdf>`__]
#

pitch_feature = F.compute_kaldi_pitch(SPEECH_WAVEFORM, SAMPLE_RATE)
pitch, nfcc = pitch_feature[..., 0], pitch_feature[..., 1]

######################################################################
#

def plot_kaldi_pitch(waveform, sr, pitch, nfcc):
    _, axis = plt.subplots(1, 1)
    axis.set_title("Kaldi Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sr
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    ln1 = axis.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")
    axis.set_ylim((-1.3, 1.3))

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, nfcc.shape[1])
    ln2 = axis2.plot(time_axis, nfcc[0], linewidth=2, label="NFCC", color="blue", linestyle="--")

    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]
    axis.legend(lns, labels, loc=0)
    plt.show(block=False)


plot_kaldi_pitch(SPEECH_WAVEFORM, SAMPLE_RATE, pitch, nfcc)
