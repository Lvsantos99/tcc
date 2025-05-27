import os
import shutil
import random
from pathlib import Path

# Caminhos base
BASE_IMAGES = Path("images/train")
BASE_LABELS = Path("labels/train")
DIR_IMAGES = Path("images")
DIR_LABELS = Path("labels")

# Destinos
TRAIN_IMG = DIR_IMAGES / "train"
VAL_IMG = DIR_IMAGES / "val"
TRAIN_LBL = DIR_LABELS / "train"
VAL_LBL = DIR_LABELS / "val"

# Criar pastas de destino
for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    d.mkdir(parents=True, exist_ok=True)

# Coleta todos os arquivos .jpg
imagens = sorted([f for f in BASE_IMAGES.glob("*.jpg")])
random.shuffle(imagens)

# Divide 80/20
divisao = int(0.8 * len(imagens))
imagens_train = imagens[:divisao]
imagens_val = imagens[divisao:]

def mover(imagens_lista, destino_img, destino_lbl):
    for img_path in imagens_lista:
        nome = img_path.name
        nome_txt = img_path.with_suffix(".txt").name

        # Move imagem
        shutil.move(str(img_path), destino_img / nome)

        # Move label correspondente
        lbl_path = BASE_LABELS / nome_txt
        if lbl_path.exists():
            shutil.move(str(lbl_path), destino_lbl / nome_txt)

# Realiza a divisão
mover(imagens_train, TRAIN_IMG, TRAIN_LBL)
mover(imagens_val, VAL_IMG, VAL_LBL)

print(f"✅ {len(imagens_train)} imagens para treino")
print(f"✅ {len(imagens_val)} imagens para validação")

# Cria data.yaml
yaml_content = f"""train: {TRAIN_IMG.resolve()}
val: {VAL_IMG.resolve()}

nc: 1
names: ["jogador"]
"""

with open("data.yaml", "w") as f:
    f.write(yaml_content)

print("📄 Arquivo 'data.yaml' criado com sucesso!")
