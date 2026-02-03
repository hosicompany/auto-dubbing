#!/usr/bin/env python3
"""
Smart Dubbing Pipeline v3
- VAD로 사람 음성 구간만 감지
- 배경음악/효과음 유지
- 음성 구간에만 더빙 적용
"""

import os
import sys
import json
import subprocess
import tempfile
import argparse
from pathlib import Path
import torch
from openai import OpenAI
import requests
import time

# ============ 설정 ============
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_7b0a163f718c23222429625faebe9dabf428825ebc36d6c2")
ELEVENLABS_VOICE_ID = "pFZP5JQG7iQjIQuC4Bku"  # Lily (여성) - 기본 목소리
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 사용 가능한 ElevenLabs 목소리 (화자 분리용)
ELEVENLABS_VOICES = {
    "male_1": "pNInz6obpgDQGcFmaJgB",    # Adam (남성)
    "male_2": "VR6AewLTigWG4xSOukaG",    # Arnold (남성)
    "female_1": "pFZP5JQG7iQjIQuC4Bku",  # Lily (여성)
    "female_2": "21m00Tcm4TlvDq8ikWAM",  # Rachel (여성)
}

# Time-stretch 허용 범위
STRETCH_TOLERANCE = 0.25
MIN_STRETCH_RATIO = 1 - STRETCH_TOLERANCE
MAX_STRETCH_RATIO = 1 + STRETCH_TOLERANCE

# VAD 설정
VAD_THRESHOLD = 0.5  # 음성 감지 임계값
MIN_SPEECH_DURATION = 0.5  # 최소 음성 길이 (초)
MIN_SILENCE_DURATION = 0.3  # 최소 무음 길이 (초)


def clone_voice_elevenlabs(audio_samples: list, voice_name: str = "cloned_voice") -> str:
    """ElevenLabs에 음성 클로닝 요청 (voice_id 반환)"""
    print(f"[클로닝] 음성 클로닝 중: {voice_name}")
    
    url = "https://api.elevenlabs.io/v1/voices/add"
    
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    # 파일 준비
    files = []
    for i, sample_path in enumerate(audio_samples):
        files.append(
            ("files", (f"sample_{i}.mp3", open(sample_path, "rb"), "audio/mpeg"))
        )
    
    data = {
        "name": voice_name,
        "description": f"Cloned voice for dubbing - {voice_name}"
    }
    
    try:
        response = requests.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        
        result = response.json()
        voice_id = result.get("voice_id")
        print(f"[클로닝] 완료! Voice ID: {voice_id}")
        return voice_id
        
    except Exception as e:
        print(f"[클로닝] 실패: {e}")
        return None
    finally:
        # 파일 핸들 닫기
        for _, (_, f, _) in files:
            f.close()


def extract_voice_sample(audio_path: str, segments: list, output_path: str, 
                         target_duration: float = 30.0) -> str:
    """음성 샘플 추출 (클로닝용, 약 30초)"""
    print(f"[샘플] 음성 샘플 추출 중 (목표: {target_duration}초)")
    
    # 긴 세그먼트들 선택 (품질 좋은 샘플)
    sorted_segs = sorted(segments, key=lambda x: x['duration'], reverse=True)
    
    selected = []
    total_duration = 0
    
    for seg in sorted_segs:
        if total_duration >= target_duration:
            break
        selected.append(seg)
        total_duration += seg['duration']
    
    if not selected:
        return None
    
    # 선택된 세그먼트들 합치기
    filter_parts = []
    for i, seg in enumerate(selected):
        filter_parts.append(f"[0:a]atrim={seg['start']}:{seg['end']},asetpts=PTS-STARTPTS[a{i}]")
    
    concat_inputs = "".join([f"[a{i}]" for i in range(len(selected))])
    filter_parts.append(f"{concat_inputs}concat=n={len(selected)}:v=0:a=1[out]")
    
    filter_complex = ";".join(filter_parts)
    
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "libmp3lame", "-q:a", "2",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"[샘플] 추출 완료: {output_path} ({total_duration:.1f}초)")
    return output_path


def send_telegram_notification(message: str, chat_id: str, bot_token: str, 
                               file_path: str = None) -> bool:
    """텔레그램으로 알림 전송 (파일 첨부 가능)"""
    try:
        if file_path and os.path.exists(file_path):
            # 파일과 함께 전송
            url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
            with open(file_path, 'rb') as f:
                response = requests.post(url, data={
                    "chat_id": chat_id,
                    "caption": message
                }, files={"document": f})
        else:
            # 텍스트만 전송
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            response = requests.post(url, json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            })
        return response.status_code == 200
    except Exception as e:
        print(f"[알림] 텔레그램 전송 실패: {e}")
        return False


def is_youtube_url(url: str) -> bool:
    """유튜브 URL인지 확인"""
    youtube_patterns = [
        'youtube.com/watch',
        'youtu.be/',
        'youtube.com/shorts/'
    ]
    return any(p in url for p in youtube_patterns)


def download_youtube(url: str, output_dir: str) -> str:
    """유튜브 영상 다운로드"""
    print(f"[YouTube] 다운로드 중: {url}")
    
    output_template = os.path.join(output_dir, "%(title).50s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "-o", output_template,
        "--no-playlist",
        "--print", "after_move:filepath",
        url
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    downloaded_path = result.stdout.strip().split('\n')[-1]
    
    print(f"[YouTube] 다운로드 완료: {os.path.basename(downloaded_path)}")
    return downloaded_path


def extract_audio_wav(video_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """영상에서 오디오 추출 (WAV, 모노)"""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def detect_speech_segments(audio_path: str) -> list:
    """Silero VAD로 사람 음성 구간 감지"""
    print("[VAD] 음성 구간 감지 중...")
    
    import scipy.io.wavfile as wavfile
    import numpy as np
    
    # Silero VAD 모델 로드
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False,
        trust_repo=True
    )
    
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    
    # scipy로 직접 오디오 로드 (torchaudio 우회)
    sample_rate, audio_data = wavfile.read(audio_path)
    
    # int16 → float32 변환 (-1 ~ 1 범위)
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    
    # 모노로 변환
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # 리샘플링 (16000Hz 필요)
    if sample_rate != 16000:
        from scipy import signal
        num_samples = int(len(audio_data) * 16000 / sample_rate)
        audio_data = signal.resample(audio_data, num_samples)
    
    # numpy → torch tensor
    wav = torch.from_numpy(audio_data).float()
    
    # 음성 구간 감지
    speech_timestamps = get_speech_timestamps(
        wav, 
        model,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
        min_silence_duration_ms=int(MIN_SILENCE_DURATION * 1000),
        sampling_rate=16000
    )
    
    # 샘플 → 초 변환
    segments = []
    for ts in speech_timestamps:
        start_sec = ts['start'] / 16000
        end_sec = ts['end'] / 16000
        segments.append({
            'start': round(start_sec, 2),
            'end': round(end_sec, 2),
            'duration': round(end_sec - start_sec, 2)
        })
    
    print(f"[VAD] {len(segments)}개 음성 구간 감지")
    if segments:
        print(f"      첫 음성 시작: {segments[0]['start']}초")
    
    return segments


def extract_audio_segment(audio_path: str, start: float, end: float, output_path: str) -> str:
    """오디오에서 특정 구간 추출"""
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ss", str(start), "-to", str(end),
        "-acodec", "libmp3lame", "-q:a", "2",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def transcribe_audio_segment(audio_path: str, client: OpenAI) -> str:
    """Whisper로 오디오 세그먼트 텍스트 변환"""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    return response.strip()


def transcribe_speech_segments(audio_path: str, speech_segments: list, temp_dir: str) -> list:
    """각 음성 구간을 Whisper로 텍스트 변환"""
    print("[Whisper] 음성 구간 텍스트 변환 중...")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    results = []
    for i, seg in enumerate(speech_segments):
        print(f"  변환 중: {i+1}/{len(speech_segments)}", end="\r")
        
        # 구간 추출
        seg_audio_path = os.path.join(temp_dir, f"speech_seg_{i:04d}.mp3")
        extract_audio_segment(audio_path, seg['start'], seg['end'], seg_audio_path)
        
        # Whisper로 텍스트 변환
        text = transcribe_audio_segment(seg_audio_path, client)
        
        if text:  # 텍스트가 있는 경우만
            results.append({
                **seg,
                'text': text,
                'audio_path': seg_audio_path
            })
    
    print(f"\n[Whisper] {len(results)}개 세그먼트 텍스트 변환 완료")
    return results


def translate_segments(segments: list, target_lang: str = "Korean", tone: str = "formal") -> list:
    """세그먼트 번역 (길이 힌트 + 말투 포함)"""
    print(f"[번역] 번역 중... (말투: {tone})")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # 말투 설명
    tone_instructions = {
        "formal": "Use polite/formal Korean (존댓말, ~습니다/~요 endings). Suitable for professional or educational content.",
        "casual": "Use casual Korean (반말, ~해/~야 endings). Suitable for friendly, informal content.",
        "narration": "Use narration style Korean (나레이션체, ~다/~했다 endings). Suitable for documentaries or storytelling."
    }
    
    tone_desc = tone_instructions.get(tone, tone_instructions["formal"])
    
    # 번역할 데이터 준비
    texts_with_duration = []
    for i, seg in enumerate(segments):
        texts_with_duration.append({
            "id": i,
            "text": seg["text"],
            "duration_sec": seg["duration"]
        })
    
    prompt = f"""You are a professional dubbing translator. Translate the following segments to {target_lang}.

TONE/STYLE: {tone_desc}

IMPORTANT RULES:
1. Each segment has a duration in seconds - the translation should be speakable in roughly that time
2. If the direct translation is too long, paraphrase or shorten it naturally
3. If the direct translation is too short, you can slightly expand it
4. Prioritize natural speech over literal translation
5. Keep the meaning and tone intact
6. MUST follow the specified tone/style consistently

Return a JSON array with "id" and "translation" for each segment.

Segments:
{json.dumps(texts_with_duration, ensure_ascii=False, indent=2)}

Return ONLY the JSON array, no markdown or other text."""

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content
    
    # JSON 추출
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    result = json.loads(content.strip())
    translation_map = {item["id"]: item["translation"] for item in result}
    
    for i, seg in enumerate(segments):
        seg["translated"] = translation_map.get(i, seg["text"])
    
    print(f"[번역] 완료")
    return segments


def generate_tts(text: str, output_path: str, voice_id: str = None) -> float:
    """ElevenLabs TTS 생성"""
    vid = voice_id or ELEVENLABS_VOICE_ID
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    return get_audio_duration(output_path)


def get_audio_duration(audio_path: str) -> float:
    """오디오 파일 길이 측정"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def time_stretch_audio(input_path: str, output_path: str, ratio: float) -> str:
    """오디오 속도 조절"""
    ratio = max(0.5, min(2.0, ratio))
    
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter:a", f"atempo={ratio}",
        "-vn", output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def process_tts_with_sync(segments: list, temp_dir: str, voice_id: str = None) -> list:
    """TTS 생성 + 싱크 조정"""
    print("[TTS] 음성 생성 및 싱크 조정 중...")
    
    processed = []
    
    for i, seg in enumerate(segments):
        print(f"  처리 중: {i+1}/{len(segments)}", end="\r")
        
        tts_path = os.path.join(temp_dir, f"tts_{i:04d}.mp3")
        stretched_path = os.path.join(temp_dir, f"stretched_{i:04d}.mp3")
        
        # TTS 생성
        tts_duration = generate_tts(seg["translated"], tts_path, voice_id)
        
        # 타겟 길이
        target_duration = seg["duration"]
        
        # 비율 계산
        stretch_ratio = target_duration / tts_duration if tts_duration > 0 else 1.0
        
        # Time-stretch 적용 여부 결정
        final_path = tts_path
        
        if MIN_STRETCH_RATIO <= stretch_ratio <= MAX_STRETCH_RATIO:
            if abs(stretch_ratio - 1.0) > 0.05:
                time_stretch_audio(tts_path, stretched_path, stretch_ratio)
                final_path = stretched_path
        else:
            status = "빠름" if stretch_ratio < 1 else "느림"
            print(f"\n  [!] 세그먼트 {i}: 범위 초과 ({status}) - TTS: {tts_duration:.1f}s, 타겟: {target_duration:.1f}s")
        
        processed.append({
            **seg,
            "tts_path": final_path,
            "tts_duration": tts_duration,
            "final_duration": get_audio_duration(final_path)
        })
    
    print(f"\n[TTS] {len(processed)}개 세그먼트 처리 완료")
    return processed


def mix_dubbed_audio(original_audio: str, segments: list, output_path: str,
                     original_volume: float = 0.15, auto_ducking: bool = True) -> str:
    """원본 오디오 + 더빙 믹싱 (자동 볼륨 조절 지원)"""
    print("[믹싱] 오디오 믹싱 중...")
    
    if not segments:
        subprocess.run(["ffmpeg", "-y", "-i", original_audio, output_path], 
                      capture_output=True, check=True)
        return output_path
    
    # 오디오 길이 확인
    total_duration = get_audio_duration(original_audio)
    
    if auto_ducking:
        print("[믹싱] 자동 볼륨 조절 (더빙 구간 감지)...")
        # 더빙 구간에서만 볼륨 낮추기, 나머지는 유지
        ducking_volume = original_volume  # 더빙 중 볼륨 (낮음)
        normal_volume = min(0.7, original_volume * 4)  # 더빙 없을 때 볼륨 (높음)
        
        # 볼륨 변화 구간 생성
        volume_expr_parts = []
        
        for seg in segments:
            start = seg["start"]
            end = seg["end"]
            # 더빙 구간: 낮은 볼륨
            volume_expr_parts.append(f"between(t,{start},{end})*{ducking_volume}")
        
        # 더빙 없는 구간: 높은 볼륨
        # volume = normal_volume * (1 - any_ducking) + ducking_volume * any_ducking
        ducking_expr = "+".join(volume_expr_parts) if volume_expr_parts else "0"
        volume_filter = f"volume='if(gt({ducking_expr},0),{ducking_volume},{normal_volume})':eval=frame"
    else:
        volume_filter = f"volume={original_volume}"
    
    # 복잡한 필터 체인 구성
    inputs = ["-i", original_audio]
    filter_parts = []
    
    # 각 TTS 파일 입력 추가
    for i, seg in enumerate(segments):
        inputs.extend(["-i", seg["tts_path"]])
    
    # 원본 오디오 볼륨 조절 (자동 덕킹 적용)
    filter_parts.append(f"[0:a]{volume_filter}[orig]")
    
    # 각 더빙 세그먼트에 딜레이 적용
    overlay_inputs = ["[orig]"]
    for i, seg in enumerate(segments):
        delay_ms = int(seg["start"] * 1000)
        filter_parts.append(
            f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[dub{i}]"
        )
        overlay_inputs.append(f"[dub{i}]")
    
    # 모든 트랙 믹스
    filter_parts.append(
        f"{''.join(overlay_inputs)}amix=inputs={len(segments)+1}:duration=longest:normalize=0[out]"
    )
    
    filter_complex = ";".join(filter_parts)
    
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:a", "libmp3lame", "-q:a", "2",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    print("[믹싱] 완료")
    return output_path


def replace_video_audio(video_path: str, audio_path: str, output_path: str):
    """영상의 오디오 교체"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def format_srt_time(seconds: float) -> str:
    """초를 SRT 시간 형식으로 변환 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments: list, output_path: str, dual: bool = False) -> str:
    """SRT 자막 파일 생성 (이중 자막 지원)"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start_time = format_srt_time(seg["start"])
            end_time = format_srt_time(seg["end"])
            
            if dual:
                # 이중 자막: 영어 원본 + 한국어 번역
                original = seg.get("text", "")
                translated = seg.get("translated", "")
                text = f"{original}\n{translated}"
            else:
                # 단일 자막: 번역만
                text = seg.get("translated", seg.get("text", ""))
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
    
    return output_path


def burn_subtitles(video_path: str, srt_path: str, output_path: str,
                   fontsize: int = 24, fontcolor: str = "white",
                   outline: int = 2, font: str = "NanumGothic") -> str:
    """자막을 영상에 하드코딩 (burn-in)"""
    print("[자막] 자막 합성 중...")
    
    # Windows 경로 이스케이프
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
    
    # 자막 스타일
    style = f"FontSize={fontsize},FontName={font},PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline={outline},Shadow=1"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles='{srt_escaped}':force_style='{style}'",
        "-c:a", "copy",
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] 자막 합성 실패, 기본 스타일로 재시도...")
        # 폰트 없을 경우 기본 스타일로 재시도
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles='{srt_escaped}'",
            "-c:a", "copy",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    
    print("[자막] 완료")
    return output_path


def process_single_video(input_path: Path, output_path: str, args) -> bool:
    """단일 영상 처리 (배치용)"""
    start_time = time.time()
    
    try:
        print(f"\n{'='*50}")
        print(f"처리 중: {input_path.name}")
        print(f"{'='*50}\n")
        
        # 임시 디렉토리
        if args.keep_temp:
            temp_dir = f"./temp_dub_{input_path.stem}"
            os.makedirs(temp_dir, exist_ok=True)
        else:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name
        
        try:
            # 1. 오디오 추출
            print("[1/7] 오디오 추출 중...")
            audio_wav = os.path.join(temp_dir, "original.wav")
            extract_audio_wav(str(input_path), audio_wav)
            
            # 2. VAD
            print("[2/7] VAD 음성 구간 감지 중...")
            speech_segments = detect_speech_segments(audio_wav)
            
            if not speech_segments:
                print("[!] 음성이 감지되지 않았습니다!")
                return False
            
            # 3. Whisper
            print("[3/7] 음성 구간 텍스트 변환 중...")
            audio_mp3 = os.path.join(temp_dir, "original.mp3")
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_wav,
                "-acodec", "libmp3lame", "-q:a", "2", audio_mp3
            ], capture_output=True, check=True)
            
            segments = transcribe_speech_segments(audio_mp3, speech_segments, temp_dir)
            
            if not segments:
                print("[!] 텍스트 변환 결과가 없습니다!")
                return False
            
            # 4. 번역
            print("[4/7] 번역 중...")
            segments = translate_segments(segments, args.lang, args.tone)
            
            # 5. TTS (음성 클로닝 또는 선택한 목소리)
            print("[5/7] TTS 생성 중...")
            
            voice_id = None
            if hasattr(args, 'clone_voice') and args.clone_voice:
                # 음성 클로닝
                print("[클로닝] 원본 화자 음성 클로닝 시작...")
                sample_path = os.path.join(temp_dir, "voice_sample.mp3")
                extract_voice_sample(audio_mp3, speech_segments, sample_path)
                voice_id = clone_voice_elevenlabs([sample_path], f"cloned_{input_path.stem}")
                if not voice_id:
                    print("[클로닝] 실패, 기본 목소리 사용")
            elif hasattr(args, 'voice') and args.voice:
                # 선택한 목소리
                voice_id = ELEVENLABS_VOICES.get(args.voice)
                print(f"[TTS] 선택한 목소리: {args.voice}")
            
            segments = process_tts_with_sync(segments, temp_dir, voice_id)
            
            # 6. 자막
            srt_path = os.path.join(temp_dir, "subtitle_ko.srt")
            generate_srt(segments, srt_path, dual=args.dual_sub)
            
            output_srt = f"{Path(output_path).stem}.srt"
            import shutil
            shutil.copy(srt_path, output_srt)
            
            # 7. 합성
            if args.subtitle_only:
                print("[7/7] 자막만 합성 중...")
                burn_subtitles(str(input_path), srt_path, output_path, fontsize=args.fontsize)
            else:
                print("[7/7] 최종 합성 중...")
                mixed_audio = os.path.join(temp_dir, "mixed.mp3")
                auto_ducking = not args.no_ducking
                mix_dubbed_audio(audio_mp3, segments, mixed_audio, args.original_volume, auto_ducking)
                
                dubbed_video = os.path.join(temp_dir, "dubbed_no_sub.mp4")
                replace_video_audio(str(input_path), mixed_audio, dubbed_video)
                
                if args.subtitle or args.dual_sub:
                    burn_subtitles(dubbed_video, srt_path, output_path, fontsize=args.fontsize)
                else:
                    shutil.copy(dubbed_video, output_path)
            
            elapsed = time.time() - start_time
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            
            print(f"[OK] 완료: {output_path}")
            print(f"[OK] 소요시간: {elapsed_min}분 {elapsed_sec}초")
            
            # 텔레그램 알림
            if hasattr(args, 'notify') and args.notify:
                notify_msg = f"🎬 <b>더빙 완료!</b>\n\n"
                notify_msg += f"📁 파일: {input_path.name}\n"
                notify_msg += f"⏱ 소요시간: {elapsed_min}분 {elapsed_sec}초\n"
                notify_msg += f"✅ 출력: {Path(output_path).name}"
                
                send_telegram_notification(
                    notify_msg,
                    args.telegram_chat_id,
                    args.telegram_bot_token
                )
                print("[알림] 텔레그램 알림 전송 완료")
            
            return True
            
        finally:
            if not args.keep_temp and 'temp_dir_obj' in dir():
                temp_dir_obj.cleanup()
                
    except Exception as e:
        print(f"[ERROR] {input_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Smart Dubbing Pipeline v3 (VAD)")
    parser.add_argument("input", nargs="?", help="입력 영상 파일 또는 유튜브 URL")
    parser.add_argument("-o", "--output", help="출력 파일명")
    parser.add_argument("--batch", help="배치 처리: 폴더 경로 지정")
    parser.add_argument("--batch-output", help="배치 출력 폴더 (기본: input_dubbed)")
    parser.add_argument("-l", "--lang", default="Korean", help="타겟 언어")
    parser.add_argument("--original-volume", type=float, default=0.15, help="원본 오디오 볼륨")
    parser.add_argument("--subtitle", action="store_true", help="자막 포함")
    parser.add_argument("--subtitle-only", action="store_true", help="자막만 생성 (더빙 없이)")
    parser.add_argument("--dual-sub", action="store_true", help="이중 자막 (영어+한국어)")
    parser.add_argument("--fontsize", type=int, default=24, help="자막 폰트 크기")
    parser.add_argument("--tone", choices=["formal", "casual", "narration"], default="formal",
                        help="말투 선택: formal(존댓말), casual(반말), narration(나레이션체)")
    parser.add_argument("--auto-ducking", action="store_true", default=True,
                        help="자동 볼륨 조절 (더빙 구간에서 배경음 낮춤)")
    parser.add_argument("--no-ducking", action="store_true",
                        help="자동 볼륨 조절 비활성화")
    parser.add_argument("--notify", action="store_true", help="완료 시 텔레그램 알림")
    parser.add_argument("--clone-voice", action="store_true", help="원본 화자 음성 클로닝")
    parser.add_argument("--voice", choices=["male_1", "male_2", "female_1", "female_2"],
                        help="TTS 목소리 선택 (클로닝 안 할 때)")
    parser.add_argument("--telegram-chat-id", default="6329826367", help="텔레그램 Chat ID")
    parser.add_argument("--telegram-bot-token", 
                        default="8293841489:AAE6XG6x5v0Prgs0bqsVMlK9Fe_K46ESWng",
                        help="텔레그램 Bot Token")
    parser.add_argument("--keep-temp", action="store_true", help="임시 파일 유지")
    args = parser.parse_args()
    
    # 배치 처리 모드
    if args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.exists():
            print(f"[!] 폴더를 찾을 수 없습니다: {batch_dir}")
            sys.exit(1)
        
        # 영상 파일 찾기
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        video_files = [f for f in batch_dir.iterdir() 
                      if f.suffix.lower() in video_extensions]
        
        if not video_files:
            print(f"[!] 영상 파일이 없습니다: {batch_dir}")
            sys.exit(1)
        
        # 출력 폴더
        output_dir = Path(args.batch_output) if args.batch_output else batch_dir / "dubbed"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"배치 처리 시작")
        print(f"{'='*50}")
        print(f"입력 폴더: {batch_dir}")
        print(f"출력 폴더: {output_dir}")
        print(f"영상 개수: {len(video_files)}개")
        print(f"{'='*50}\n")
        
        success = 0
        failed = 0
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] {video_file.name}")
            output_path = output_dir / f"{video_file.stem}_dubbed_ko{video_file.suffix}"
            
            if process_single_video(video_file, str(output_path), args):
                success += 1
            else:
                failed += 1
        
        print(f"\n{'='*50}")
        print(f"배치 처리 완료!")
        print(f"성공: {success}개 / 실패: {failed}개")
        print(f"{'='*50}\n")
        
        # 배치 완료 알림
        if args.notify:
            notify_msg = f"📦 <b>배치 처리 완료!</b>\n\n"
            notify_msg += f"📁 폴더: {batch_dir.name}\n"
            notify_msg += f"✅ 성공: {success}개\n"
            notify_msg += f"❌ 실패: {failed}개"
            
            send_telegram_notification(
                notify_msg,
                args.telegram_chat_id,
                args.telegram_bot_token
            )
            print("[알림] 텔레그램 알림 전송 완료")
        
        return
    
    # 단일 파일 모드 - input 필수
    if not args.input:
        print("[!] 입력 파일을 지정해주세요.")
        print("사용법: python smart_dub_v3.py <영상파일 또는 유튜브URL>")
        print("배치:   python smart_dub_v3.py --batch <폴더경로>")
        sys.exit(1)
    
    # 유튜브 URL 처리
    if is_youtube_url(args.input):
        print(f"\n[YouTube] URL 감지됨!")
        temp_download_dir = "./temp_youtube"
        os.makedirs(temp_download_dir, exist_ok=True)
        downloaded_file = download_youtube(args.input, temp_download_dir)
        input_path = Path(downloaded_file)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[!] 파일을 찾을 수 없습니다: {input_path}")
            sys.exit(1)
    
    output_path = args.output or f"{input_path.stem}_dubbed_ko{input_path.suffix}"
    
    # 단일 파일 처리
    success = process_single_video(input_path, output_path, args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
