import os
import cv2
from ultralytics import YOLO

def process_video(video_path, model, step=3, resize=None, visualize=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Erro ao abrir: {video_path}")
        return

    frame_id = 0
    print(f"[→] Processando: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            if resize:
                frame = cv2.resize(frame, resize)

            results = model(frame)[0]  # YOLOv8
            if visualize:
                annotated = results.plot()
                cv2.imshow("YOLOv8", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_id += 1

    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    print(f"[✓] Finalizado: {video_path}")

def process_all_videos(root_dir="videos", step=3, resize=None, visualize=True):
    model = YOLO("yolov8n.pt")  # Troque para yolov8s.pt ou outro se quiser
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith((".mkv", ".mp4", ".avi")):
                full_path = os.path.join(dirpath, file)
                process_video(full_path, model, step, resize, visualize)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Processar todos os vídeos de uma pasta (com subpastas)")
    parser.add_argument("--input", type=str, default="videos", help="Diretório com vídeos")
    parser.add_argument("--step", type=int, default=3, help="Intervalo entre frames")
    parser.add_argument("--resize", type=int, nargs=2, help="Redimensionar para (W H)")
    parser.add_argument("--novis", action="store_true", help="Desativar visualização")

    args = parser.parse_args()
    size = tuple(args.resize) if args.resize else None
    process_all_videos(args.input, args.step, size, not args.novis)
