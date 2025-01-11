import os
import cv2
import numpy as np

# Caminho das imagens
caminho_original = "imagens/original"
caminho_original_cinza = "imagens/original_cinza"
caminho_preprocessamento = "imagens/filtragem"
caminho_segmentadas = "imagens/segmentadas"

# Função para converter imagem para escala de cinza
def converter_para_cinza(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Função para aplicar filtro de passa-baixa (média)
def aplicar_filtro_passabaixa(image, kernel_size=(3, 3)):
    w_low = np.ones(kernel_size, dtype=float) / 9
    return cv2.filter2D(image, -1, w_low)

# Função para segmentar a imagem usando Otsu
def segmentar_com_otsu(image, nivel_branco=255, ajuste_limiar=False):
    """
    Aplicação do método de segmentação Otsu com controle do nível de branco e possibilidade de ajuste do limiar:
    
    Parâmetros:
    - image: A imagem de entrada (em escala de cinza).
    - nivel_branco: O valor do nível de branco desejado para os pixels acima do limiar Otsu.
    - ajuste_limiar: Se True, permite ajustar o limiar manualmente.
    
    Retorna:
    - Uma imagem binária com a segmentação ajustada.
    """
    # Aplica o limiar de Otsu (automático)
    _, segmentada_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if ajuste_limiar:
        # Aqui você pode ajustar o limiar manualmente
        limiar_manual = 100  # Ajuste esse valor conforme necessário
        _, segmentada_otsu = cv2.threshold(image, limiar_manual, 255, cv2.THRESH_BINARY)
    
    # Ajusta o nível de branco
    segmentada = np.where(segmentada_otsu == 255, nivel_branco, 0).astype(np.uint8)
    
    return segmentada

# 1. Converter imagens para cinza e aplicar filtro de passa-baixa
for arquivo in os.listdir(caminho_original):
    if arquivo.endswith('.png'):
        caminho_imagem = os.path.join(caminho_original, arquivo)
        imagem = cv2.imread(caminho_imagem)
        
        # Converter para escala de cinza e salvar no arquivo original_cinza
        cinza = converter_para_cinza(imagem)
        caminho_saida_cinza = os.path.join(caminho_original_cinza, arquivo)
        cv2.imwrite(caminho_saida_cinza, cinza)
        
        # Aplicar filtro de passa-baixa à imagem cinza
        imagem_filtrada = aplicar_filtro_passabaixa(cinza)
        caminho_saida_filtrada = os.path.join(caminho_preprocessamento, arquivo)
        cv2.imwrite(caminho_saida_filtrada, imagem_filtrada)

# 2. Aplicar segmentação com Otsu nas imagens preprocessadas
for arquivo in os.listdir(caminho_preprocessamento):
    if arquivo.endswith('.png'):
        caminho_imagem_filtrada = os.path.join(caminho_preprocessamento, arquivo)
        imagem_preprocessada = cv2.imread(caminho_imagem_filtrada, cv2.IMREAD_GRAYSCALE)
        
        # Aplicar segmentação com Otsu
        imagem_segmentada = segmentar_com_otsu(imagem_preprocessada, ajuste_limiar=True)
        caminho_saida_segmentada = os.path.join(caminho_segmentadas, arquivo)
        cv2.imwrite(caminho_saida_segmentada, imagem_segmentada)

print("Segmentação com Otsu concluída!")
