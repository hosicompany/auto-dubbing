# 🎬 Auto-Dubbing Pipeline

영어 영상을 한국어 더빙 + 자막 영상으로 자동 변환하는 AI 파이프라인

## ✨ 주요 기능

- **VAD (Voice Activity Detection)** - 사람 음성 구간만 정확히 감지
- **자동 번역** - GPT를 활용한 자연스러운 한국어 번역
- **TTS 더빙** - ElevenLabs를 활용한 고품질 한국어 음성 생성
- **자막 합성** - 영상에 한국어 자막 자동 삽입
- **유튜브 지원** - URL만 입력하면 자동 다운로드 + 더빙
- **배치 처리** - 여러 영상 일괄 처리
- **텔레그램 알림** - 처리 완료 시 알림

## 🚀 설치

### 필수 요구사항

- Python 3.10+
- FFmpeg
- yt-dlp

### Python 패키지

```bash
pip install openai requests torch scipy
```

### API 키 설정

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
```

## 📖 사용법

### 기본 사용

```bash
# 로컬 영상 파일
python smart_dub_v3.py video.mp4

# 유튜브 URL
python smart_dub_v3.py "https://youtube.com/watch?v=xxx"

# 자막 포함
python smart_dub_v3.py video.mp4 --subtitle
```

### 고급 옵션

```bash
# 말투 선택
python smart_dub_v3.py video.mp4 --tone formal    # 존댓말 (기본)
python smart_dub_v3.py video.mp4 --tone casual    # 반말
python smart_dub_v3.py video.mp4 --tone narration # 나레이션체

# 이중 자막 (영어 + 한국어)
python smart_dub_v3.py video.mp4 --subtitle --dual-sub

# 배치 처리 (폴더 내 모든 영상)
python smart_dub_v3.py --batch ./videos/

# 텔레그램 알림
python smart_dub_v3.py video.mp4 --notify
```

### 전체 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--subtitle` | 자막 포함 | off |
| `--subtitle-only` | 자막만 (더빙 없이) | off |
| `--dual-sub` | 이중 자막 (영어+한국어) | off |
| `--tone` | 말투 (formal/casual/narration) | formal |
| `--fontsize` | 자막 폰트 크기 | 24 |
| `--original-volume` | 원본 오디오 볼륨 | 0.15 |
| `--no-ducking` | 자동 볼륨 조절 끄기 | off |
| `--batch` | 배치 처리 폴더 | - |
| `--notify` | 텔레그램 알림 | off |
| `--keep-temp` | 임시 파일 유지 | off |

## 🔧 파이프라인 구조

```
입력 영상
    ↓
[1] 오디오 추출 (FFmpeg)
    ↓
[2] VAD 음성 구간 감지 (Silero VAD)
    ↓
[3] 음성 → 텍스트 (Whisper API)
    ↓
[4] 번역 (GPT)
    ↓
[5] TTS 음성 생성 (ElevenLabs)
    ↓
[6] 자막 생성 (SRT)
    ↓
[7] 오디오 믹싱 + 영상 합성
    ↓
출력 (더빙 영상 + SRT 파일)
```

## 💰 예상 비용

| 영상 길이 | 예상 비용 |
|-----------|----------|
| 3분 | ~$0.40 |
| 10분 | ~$1.30 |
| 20분 | ~$2.50 |

> 대부분 ElevenLabs TTS 비용 (전체의 ~80%)

## 📁 출력 파일

```
video.mp4 (원본)
    ↓
video_dubbed_ko.mp4  # 더빙된 영상
video_dubbed_ko.srt  # 한국어 자막 파일
```

## 🛠️ 기술 스택

- **음성 인식**: OpenAI Whisper API
- **VAD**: Silero VAD
- **번역**: GPT-5-nano
- **TTS**: ElevenLabs (eleven_multilingual_v2)
- **영상 처리**: FFmpeg

## 📜 라이선스

MIT License

## 🦊 만든이

로티 (Lottie) - AI Assistant
