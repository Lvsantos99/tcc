import os

image_width = 1280
image_height = 720

with open("/home/luiz/Documentos/FinalTCC/path/to/SoccerNet/tracking/train/SNMOT-061/gt/gt.txt", "r") as f:
    lines = f.readlines()

output_dir = "labels"
os.makedirs(output_dir, exist_ok=True)

for line in lines:
    frame, obj_id, x, y, w, h, _, class_id, _, _ = map(float, line.strip().split(','))

    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    w_norm = w / image_width
    h_norm = h / image_height

    label_file = os.path.join(output_dir, f"{int(frame):06d}.txt")
    with open(label_file, "a") as out_f:
        # classe 0 = jogador, 1 = bola (exemplo)
        cls = int(class_id)
        out_f.write(f"{cls} {x_center} {y_center} {w_norm} {h_norm}\n")


