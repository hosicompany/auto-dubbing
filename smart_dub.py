#!/usr/bin/env python3
"""
Smart Dubbing Pipeline v2
- 원본 타이밍에 맞춘 자연스러운 더빙
- 번역 시 길이 고려
- Time-stretch로 싱크 맞춤
"""

import os
import sys
import json
import subprocess
import tempfile
import argparse
from pathlib import Path
from openai import OpenAI
import requests

# ============ 설정 ============
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "sk_7b0a163f718c23222429625faebe9dabf428825ebc36d6c2")
ELEVENLABS_VOICE_ID = "pFZP5JQG7iQjIQuC4Bku"  # Lily
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Time-stretch 허용 범위 (예: 0.25 = ±25%)
STRETCH_TOLERANCE = 0.25
MIN_STRETCH_RATIO = 1 - STRETCH_TOLERANCE  # 0.75x
MAX_STRETCH_RATIO = 1 + STRETCH_TOLERANCE  # 1.25x


def extract_audio(video_path: str, output_path: str) -> str:
    """영상에서 오디오 추출 (MP3 형식)"""
    # Whisper API는 mp3, wav, m4a 등 지원 - mp3가 가장 호환성 좋음
    if output_path.endswith('.wav'):
        output_path = output_path.replace('.wav', '.mp3')
    
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1",
        "-q:a", "2",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def transcribe_with_timestamps(audio_path: str) -> list:
    """Whisper로 타임스탬프 포함 자막 추출"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    
    segments = []
    for seg in response.segments:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "duration": seg.end - seg.start,
            "text": seg.text.strip()
        })
    
    print(f"[*] {len(segments)}개 세그먼트 추출 완료")
    return segments


def translate_with_length_hint(segments: list, target_lang: str = "Korean") -> list:
    """길이 힌트를 포함해서 번역"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # 번역할 텍스트와 길이 정보 준비
    texts_with_duration = []
    for i, seg in enumerate(segments):
        texts_with_duration.append({
            "id": i,
            "text": seg["text"],
            "duration_sec": round(seg["duration"], 1)
        })
    
    prompt = f"""You are a professional dubbing translator. Translate the following segments to {target_lang}.

IMPORTANT RULES:
1. Each segment has a duration in seconds - the translation should be speakable in roughly that time
2. If the direct translation is too long, paraphrase or shorten it naturally
3. If the direct translation is too short, you can slightly expand it
4. Prioritize natural speech over literal translation
5. Keep the meaning and tone intact

Return a JSON array with "id" and "translation" for each segment.

Segments:
{json.dumps(texts_with_duration, ensure_ascii=False, indent=2)}

Return ONLY the JSON array, no other text."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    content = response.choices[0].message.content
    
    # JSON 추출 (코드블록 제거)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    content = content.strip()
    result = json.loads(content)
    
    # 번역 결과 매핑
    translation_map = {item["id"]: item["translation"] for item in result}
    
    for i, seg in enumerate(segments):
        seg["translated"] = translation_map.get(i, seg["text"])
    
    print(f"[*] 번역 완료")
    return segments


def generate_tts(text: str, output_path: str) -> float:
    """ElevenLabs TTS 생성 및 실제 길이 반환"""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    
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
    
    # ffprobe로 실제 길이 측정
    duration = get_audio_duration(output_path)
    return duration


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
    """오디오 속도 조절 (ratio > 1 = 느리게, ratio < 1 = 빠르게)"""
    # atempo는 0.5~2.0 범위만 지원, 그 외는 체이닝 필요
    if ratio < 0.5:
        ratio = 0.5
    elif ratio > 2.0:
        ratio = 2.0
    
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter:a", f"atempo={ratio}",
        "-vn", output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def process_segments_with_sync(segments: list, temp_dir: str) -> list:
    """각 세그먼트 TTS 생성 + 싱크 맞춤"""
    processed = []
    
    for i, seg in enumerate(segments):
        print(f"  처리 중: {i+1}/{len(segments)}", end="\r")
        
        tts_path = os.path.join(temp_dir, f"tts_{i:04d}.mp3")
        stretched_path = os.path.join(temp_dir, f"stretched_{i:04d}.mp3")
        
        # TTS 생성
        tts_duration = generate_tts(seg["translated"], tts_path)
        
        # 타겟 길이
        target_duration = seg["duration"]
        
        # 비율 계산
        if tts_duration > 0:
            stretch_ratio = target_duration / tts_duration
        else:
            stretch_ratio = 1.0
        
        # 허용 범위 내에서 time-stretch
        final_path = tts_path
        applied_stretch = 1.0
        
        if MIN_STRETCH_RATIO <= stretch_ratio <= MAX_STRETCH_RATIO:
            # 범위 내: time-stretch 적용
            if abs(stretch_ratio - 1.0) > 0.05:  # 5% 이상 차이날 때만
                time_stretch_audio(tts_path, stretched_path, stretch_ratio)
                final_path = stretched_path
                applied_stretch = stretch_ratio
        else:
            # 범위 밖: 그냥 자연스럽게 (싱크 포기)
            print(f"\n  [!] 세그먼트 {i}: 길이 차이 큼 (TTS: {tts_duration:.1f}s, 타겟: {target_duration:.1f}s)")
        
        processed.append({
            **seg,
            "tts_path": final_path,
            "tts_duration": tts_duration,
            "stretch_ratio": applied_stretch,
            "final_duration": get_audio_duration(final_path) if os.path.exists(final_path) else tts_duration
        })
    
    print(f"\n[*] {len(processed)}개 세그먼트 TTS 처리 완료")
    return processed


def mix_audio(original_audio: str, segments: list, output_path: str, 
              original_volume: float = 0.1, dub_volume: float = 1.0):
    """원본 오디오와 더빙 믹스"""
    
    # 복잡한 필터 체인 구성
    inputs = ["-i", original_audio]
    filter_parts = []
    
    # 각 TTS 파일 입력 추가
    for i, seg in enumerate(segments):
        inputs.extend(["-i", seg["tts_path"]])
    
    # 원본 오디오 볼륨 조절
    filter_parts.append(f"[0:a]volume={original_volume}[orig]")
    
    # 각 더빙 세그먼트에 딜레이 적용
    overlay_inputs = ["[orig]"]
    for i, seg in enumerate(segments):
        delay_ms = int(seg["start"] * 1000)
        filter_parts.append(
            f"[{i+1}:a]volume={dub_volume},adelay={delay_ms}|{delay_ms}[dub{i}]"
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


def main():
    parser = argparse.ArgumentParser(description="Smart Dubbing Pipeline")
    parser.add_argument("input", help="입력 영상 파일")
    parser.add_argument("-o", "--output", help="출력 파일명")
    parser.add_argument("-l", "--lang", default="Korean", help="타겟 언어 (기본: Korean)")
    parser.add_argument("--original-volume", type=float, default=0.1, help="원본 오디오 볼륨 (0-1)")
    parser.add_argument("--keep-temp", action="store_true", help="임시 파일 유지")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
    
    output_path = args.output or f"{input_path.stem}_dubbed_ko{input_path.suffix}"
    
    print(f"\n{'='*50}")
    print(f"Smart Dubbing Pipeline v2")
    print(f"{'='*50}")
    print(f"입력: {input_path}")
    print(f"출력: {output_path}")
    print(f"타겟 언어: {args.lang}")
    print(f"원본 볼륨: {args.original_volume}")
    print(f"{'='*50}\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.keep_temp:
            temp_dir = "./temp_dub"
            os.makedirs(temp_dir, exist_ok=True)
        
        # 1. 오디오 추출
        print("[1/5] 오디오 추출 중...")
        audio_path = os.path.join(temp_dir, "original.mp3")
        extract_audio(str(input_path), audio_path)
        
        # 2. Whisper 자막 추출
        print("[2/5] 자막 추출 중 (Whisper)...")
        segments = transcribe_with_timestamps(audio_path)
        
        # 3. 길이 고려 번역
        print("[3/5] 번역 중 (길이 최적화)...")
        segments = translate_with_length_hint(segments, args.lang)
        
        # 4. TTS + 싱크 맞춤
        print("[4/5] TTS 생성 및 싱크 조정 중...")
        segments = process_segments_with_sync(segments, temp_dir)
        
        # 5. 오디오 믹스
        print("[5/5] 오디오 믹싱 중...")
        mixed_audio = os.path.join(temp_dir, "mixed.mp3")
        mix_audio(audio_path, segments, mixed_audio, args.original_volume)
        
        # 최종 영상 생성
        print("[*] 최종 영상 생성 중...")
        replace_video_audio(str(input_path), mixed_audio, output_path)
        
        # 결과 리포트
        print(f"\n{'='*50}")
        print(f"[✓] 완료: {output_path}")
        print(f"{'='*50}")
        
        # 싱크 통계
        stretch_applied = sum(1 for s in segments if abs(s["stretch_ratio"] - 1.0) > 0.05)
        out_of_range = sum(1 for s in segments if s["stretch_ratio"] < MIN_STRETCH_RATIO or s["stretch_ratio"] > MAX_STRETCH_RATIO)
        
        print(f"\n📊 싱크 통계:")
        print(f"  - 총 세그먼트: {len(segments)}")
        print(f"  - Time-stretch 적용: {stretch_applied}")
        print(f"  - 범위 초과 (자연스럽게): {out_of_range}")


if __name__ == "__main__":
    main()
