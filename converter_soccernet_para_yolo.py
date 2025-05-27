import os
from pathlib import Path
import cv2

# === CONFIGURAÇÕES ===
PASTA_SEQUENCIAS = Path("/home/luiz/Documentos/FinalTCC/path/to/SoccerNet/tracking/train")
PASTA_IMAGENS_SAIDA = Path("images/train")
PASTA_LABELS_SAIDA = Path("labels/train")
PASTA_IMAGENS_SAIDA.mkdir(parents=True, exist_ok=True)
PASTA_LABELS_SAIDA.mkdir(parents=True, exist_ok=True)

# Classes SoccerNet que representam jogadores
CLASSES_JOGADOR = {1}  # Pode expandir se necessário

def processar_sequencia(pasta_seq: Path):
    gt_path = pasta_seq / "gt" / "gt.txt"
    img_dir = pasta_seq / "img1"
    nome_seq = pasta_seq.name

    if not gt_path.exists():
        print(f"❌ Sem gt.txt em {pasta_seq}")
        return

    with open(gt_path, "r") as f:
        linhas = f.readlines()

    for linha in linhas:
        partes = linha.strip().split(",")
        if len(partes) < 7:
            continue

        frame = int(partes[0])
        id_obj = int(partes[1])
        x, y, w, h = map(float, partes[2:6])
        class_id = int(partes[6])

        if class_id not in CLASSES_JOGADOR:
            continue

        nome_img = f"{frame:06d}.jpg"
        caminho_img = img_dir / nome_img
        if not caminho_img.exists():
            continue

        # Novo nome para a imagem e label
        nome_novo = f"{nome_seq}_{frame:06d}"
        img_saida = PASTA_IMAGENS_SAIDA / f"{nome_novo}.jpg"
        label_saida = PASTA_LABELS_SAIDA / f"{nome_novo}.txt"

        # Copia imagem uma vez
        if not img_saida.exists():
            img = cv2.imread(str(caminho_img))
            if img is None:
                continue
            cv2.imwrite(str(img_saida), img)
            h_img, w_img = img.shape[:2]
        else:
            img = cv2.imread(str(img_saida))
            h_img, w_img = img.shape[:2]

        # Normalizar para YOLO
        x_center = (x + w / 2) / w_img
        y_center = (y + h / 2) / h_img
        w_norm = w / w_img
        h_norm = h / h_img

        anotacao_yolo = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

        # Escreve ou adiciona anotação
        with open(label_saida, "a") as f_out:
            f_out.write(anotacao_yolo + "\n")

    print(f"✅ Processado: {nome_seq}")

# Roda para todas as sequências em "train/"
for pasta in PASTA_SEQUENCIAS.iterdir():
    if pasta.is_dir() and pasta.name.startswith("SNMOT-"):
        processar_sequencia(pasta)

print("\n🏁 Conversão completa!")
