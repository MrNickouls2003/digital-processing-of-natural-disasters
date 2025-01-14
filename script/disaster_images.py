import os
import cv2
import numpy as np
from skimage import util

# Caminho das imagens
caminho_original = "imagens/original"
caminho_original_cinza = "imagens/original_cinza"
caminho_preprocessamento = "imagens/filtragem"
caminho_segmentadas = "imagens/segmentadas"

# Criar diretórios de saída, se não existirem
os.makedirs(caminho_original_cinza, exist_ok=True)
os.makedirs(caminho_preprocessamento, exist_ok=True)
os.makedirs(caminho_segmentadas, exist_ok=True)


# Função para converter imagem para escala de cinza
def converter_para_cinza(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Função para aplicar filtro de passa-baixa (média)
def aplicar_filtro_passabaixa(image, kernel_size=(3, 3)):
    w_low = np.ones(kernel_size, dtype=float) / 9
    return cv2.filter2D(image, -1, w_low)

# Função para aplicar filtro gaussiano
def aplicar_ruido_gaussiano(image):
    return util.img_as_ubyte(util.random_noise(image, mode='gaussian'))


# Função para segmentar a imagem usando Otsu
def segmentar_com_otsu(image, nivel_branco=255, ajuste_limiar=False):
    """
    Aplicação do método de segmentação Otsu com controle do nível de branco e possibilidade de ajuste do limiar:
    
    Parâmetros:
    - image: A imagem de entrada (em escala de cinza).
    - nivel_branco: O valor do nível de branco desejado para os pixels acima do limiar Otsu.
    - ajuste_limiar: Se True, permite ajustar o limiar manualmente.
    
Análise do histograma:

Se o histograma mostra picos bem definidos (bimodal), o método de Otsu geralmente funciona bem.
Para imagens com múltiplos tons ou baixa separação, pode ser necessário um limiar manual.

Valores de limiar por tipo de desastre:

Enchentes:

Limiar: 80 - 120 (para separar água escura de terra ou vegetação).
Justificativa:
O valor foi escolhido para destacar áreas de água, que apresentam menor intensidade em relação ao solo e vegetação.

Incêndios:

Limiar: 180 - 220 (para destacar áreas brilhantes de fogo).
Justificativa:
Um limiar alto foi escolhido para capturar as regiões mais intensamente iluminadas pelo fogo.

Deslizamentos de terra:

Limiar: 100 - 150 (para separar áreas de solo exposto de vegetação).
Justificativa:
O limiar médio foi selecionado para evidenciar a diferença entre vegetação e áreas de solo recém-exposto.
    """
    # Aplica o limiar de Otsu (automático)
    _, segmentada_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if ajuste_limiar:
        # Ajustar o valor limiar manualmente
        limiar_manual = 150 
        _, segmentada_otsu = cv2.threshold(image, limiar_manual, 255, cv2.THRESH_BINARY)
    
    # Ajusta o nível de branco
    segmentada = np.where(segmentada_otsu == 255, nivel_branco, 0).astype(np.uint8)
    
    return segmentada

# 1. Converter imagens para cinza e aplicar filtros
for arquivo in os.listdir(caminho_original):
    if arquivo.endswith('.png'):
        caminho_imagem = os.path.join(caminho_original, arquivo)
        imagem = cv2.imread(caminho_imagem)
        
        # Converter para escala de cinza
        cinza = converter_para_cinza(imagem)
        caminho_saida_cinza = os.path.join(caminho_original_cinza, arquivo)
        cv2.imwrite(caminho_saida_cinza, cinza)

        # Aplicar filtro de passa-baixa
        imagem_filtrada = aplicar_filtro_passabaixa(cinza)
        
        # Aplicar ruído gaussiano na imagem filtrada
        imagem_final = aplicar_ruido_gaussiano(imagem_filtrada)
        
        # Salvar a imagem final com ambos os filtros aplicados
        caminho_saida_final = os.path.join(caminho_preprocessamento, arquivo)
        cv2.imwrite(caminho_saida_final, imagem_final)

# 2. Aplicar segmentação com Otsu nas imagens preprocessadas
for arquivo in os.listdir(caminho_preprocessamento):
    if arquivo.endswith('.png'):
        caminho_imagem_filtrada = os.path.join(caminho_preprocessamento, arquivo)
        imagem_preprocessada = cv2.imread(caminho_imagem_filtrada, cv2.IMREAD_GRAYSCALE)
        
        # Aplicar segmentação com Otsu
        imagem_segmentada = segmentar_com_otsu(imagem_preprocessada, ajuste_limiar=False)
        caminho_saida_segmentada = os.path.join(caminho_segmentadas, arquivo)
        cv2.imwrite(caminho_saida_segmentada, imagem_segmentada)

print("Filtragem e segmentacao concluidas")
