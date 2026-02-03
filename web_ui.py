#!/usr/bin/env python3
"""
Auto-Dubbing Web UI (Gradio)
"""

import gradio as gr
import subprocess
import os
import sys
from pathlib import Path

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def process_video(
    video_file,
    youtube_url,
    include_subtitle,
    dual_subtitle,
    tone,
    voice,
    clone_voice,
    notify,
    progress=gr.Progress()
):
    """영상 처리 함수"""
    
    if not video_file and not youtube_url:
        return None, None, "❌ 영상 파일 또는 유튜브 URL을 입력해주세요."
    
    # 입력 결정
    if youtube_url:
        input_source = youtube_url
        output_name = "youtube_dubbed_ko.mp4"
    else:
        input_source = video_file
        output_name = f"{Path(video_file).stem}_dubbed_ko.mp4"
    
    # 출력 경로
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    
    # 명령어 구성
    cmd = [
        sys.executable, "smart_dub_v3.py",
        input_source,
        "-o", output_path,
        "--tone", tone
    ]
    
    if include_subtitle:
        cmd.append("--subtitle")
    
    if dual_subtitle:
        cmd.append("--dual-sub")
    
    if voice and voice != "기본":
        cmd.extend(["--voice", voice])
    
    if clone_voice:
        cmd.append("--clone-voice")
    
    if notify:
        cmd.append("--notify")
    
    progress(0, desc="처리 시작...")
    
    # 환경 변수 설정
    env = os.environ.copy()
    if OPENAI_API_KEY:
        env["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    try:
        # 프로세스 실행
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        logs = []
        for line in process.stdout:
            logs.append(line.strip())
            if "[1/7]" in line:
                progress(0.1, desc="오디오 추출 중...")
            elif "[2/7]" in line:
                progress(0.2, desc="VAD 음성 감지 중...")
            elif "[3/7]" in line:
                progress(0.3, desc="음성 인식 중...")
            elif "[4/7]" in line:
                progress(0.5, desc="번역 중...")
            elif "[5/7]" in line:
                progress(0.6, desc="TTS 생성 중...")
            elif "[6/7]" in line or "자막" in line:
                progress(0.8, desc="자막 생성 중...")
            elif "[7/7]" in line:
                progress(0.9, desc="최종 합성 중...")
        
        process.wait()
        
        progress(1.0, desc="완료!")
        
        if process.returncode == 0 and os.path.exists(output_path):
            srt_path = output_path.replace(".mp4", ".srt")
            srt_file = srt_path if os.path.exists(srt_path) else None
            return output_path, srt_file, "✅ 더빙 완료!\n\n" + "\n".join(logs[-10:])
        else:
            return None, None, "❌ 처리 실패\n\n" + "\n".join(logs[-20:])
            
    except Exception as e:
        return None, None, f"❌ 오류 발생: {str(e)}"


# Gradio UI
with gr.Blocks(title="Auto-Dubbing", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎬 Auto-Dubbing Pipeline
    
    영어 영상을 한국어 더빙 + 자막 영상으로 자동 변환
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 입력")
            
            video_input = gr.File(
                label="영상 파일 업로드",
                file_types=["video"],
                type="filepath"
            )
            
            youtube_input = gr.Textbox(
                label="또는 유튜브 URL",
                placeholder="https://youtube.com/watch?v=..."
            )
            
            gr.Markdown("### ⚙️ 옵션")
            
            with gr.Row():
                subtitle_check = gr.Checkbox(label="자막 포함", value=True)
                dual_sub_check = gr.Checkbox(label="이중 자막 (영+한)")
            
            tone_select = gr.Radio(
                choices=["formal", "casual", "narration"],
                value="formal",
                label="말투",
                info="formal=존댓말, casual=반말, narration=나레이션체"
            )
            
            voice_select = gr.Dropdown(
                choices=["기본", "male_1", "male_2", "female_1", "female_2"],
                value="기본",
                label="목소리"
            )
            
            with gr.Row():
                clone_check = gr.Checkbox(label="음성 클로닝")
                notify_check = gr.Checkbox(label="텔레그램 알림")
            
            process_btn = gr.Button("🚀 더빙 시작", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📤 출력")
            
            output_video = gr.Video(label="더빙된 영상")
            output_srt = gr.File(label="자막 파일 (SRT)")
            output_log = gr.Textbox(label="처리 로그", lines=10)
    
    # 이벤트 연결
    process_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            youtube_input,
            subtitle_check,
            dual_sub_check,
            tone_select,
            voice_select,
            clone_check,
            notify_check
        ],
        outputs=[output_video, output_srt, output_log]
    )
    
    gr.Markdown("""
    ---
    ### 📖 사용법
    
    1. **영상 업로드** 또는 **유튜브 URL** 입력
    2. 옵션 선택 (자막, 말투, 목소리 등)
    3. **더빙 시작** 클릭
    4. 완료되면 영상 다운로드!
    
    ---
    Made with 🦊 by Lottie
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
