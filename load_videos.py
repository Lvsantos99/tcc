import cv2
import os

def extract_frames(video_path, output_dir, step=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    rel_path = os.path.splitext(video_path)[0]
    rel_path = rel_path.replace("videos", "").strip(os.sep)
    save_path = os.path.join(output_dir, rel_path)
    os.makedirs(save_path, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            filename = os.path.join(save_path, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[✔] {os.path.basename(video_path)}: {saved_count} frames salvos em {save_path}")

def process_all_videos(root_dir="videos", output_dir="frames", step=3):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith((".mkv", ".mp4", ".avi")):
                full_path = os.path.join(dirpath, file)
                extract_frames(full_path, output_dir, step)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extrair frames de vários vídeos em subpastas")
    parser.add_argument("--input", type=str, default="videos", help="Diretório raiz com os vídeos")
    parser.add_argument("--output", type=str, default="frames", help="Diretório de saída dos frames")
    parser.add_argument("--step", type=int, default=3, help="Intervalo entre frames")

    args = parser.parse_args()
    process_all_videos(args.input, args.output, args.step)
