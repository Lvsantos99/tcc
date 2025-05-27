import os
import cv2
from ultralytics import YOLO

from supervision import BYTETracker

def process_video(video_path, model, tracker_name, step=3, resize=None, visualize=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Erro ao abrir: {video_path}")
        return

    tracker = byte_tracker = BYTETracker(
        track_buffer=30,
        match_thresh=0.4,
        frame_rate=30,
        track_thresh=0.5,
        min_box_area=200,
        mot20=True
    ) if tracker_name == "bytetrack" else None

    frame_id = 0
    print(f"[→] Processando: {video_path} usando {tracker_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            if resize:
                frame = cv2.resize(frame, resize)

            # Detecção com YOLOv8
            results = model(frame)[0]

            # Pegando bounding boxes [x1, y1, x2, y2, conf, class]
            detections = []
            for det in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))

            # Preparando para o tracker
            if tracker_name == "deepsort":
                tracks = tracker.update_tracks(
                    detections, frame=frame
                )
            elif tracker_name == "bytetrack":
                tracks = tracker.update(detections, (frame.shape[0], frame.shape[1]), frame_id)

            # Exibe resultados
            if visualize:
                for track in tracks:
                    if tracker_name == "deepsort":
                        if not track.is_confirmed():
                            continue
                        track_id = track.track_id
                        ltrb = track.to_ltrb()
                        x1, y1, x2, y2 = map(int, ltrb)
                    elif tracker_name == "bytetrack":
                        track_id = track.track_id
                        tlwh = track.tlwh
                        x1, y1, w, h = map(int, tlwh)
                        x2 = x1 + w
                        y2 = y1 + h

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_id += 1

    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    print(f"[✓] Finalizado: {video_path}")

def process_all_videos(root_dir="videos", tracker="deepsort", step=3, resize=None, visualize=True):
    model = YOLO("yolov8n.pt")
    print(f"[📁] Procurando vídeos em {root_dir}")

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith((".mkv", ".mp4", ".avi")):
                full_path = os.path.join(dirpath, file)
                process_video(full_path, model, tracker, step, resize, visualize)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Processar vídeos com YOLOv8 + DeepSORT ou ByteTrack")
    parser.add_argument("--input", type=str, default="videos", help="Diretório dos vídeos")
    parser.add_argument("--tracker", type=str, choices=["deepsort", "bytetrack"], default="deepsort", help="Tracker para usar")
    parser.add_argument("--step", type=int, default=3, help="Intervalo de frames")
    parser.add_argument("--resize", type=int, nargs=2, help="Redimensionar (W H)")
    parser.add_argument("--novis", action="store_true", help="Sem visualização")

    args = parser.parse_args()
    size = tuple(args.resize) if args.resize else None
    process_all_videos(args.input, args.tracker, args.step, size, not args.novis)
