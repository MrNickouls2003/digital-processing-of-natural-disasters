import os
import cv2
import numpy as np
from skimage import util
import matplotlib.pyplot as plt

# Caminho das imagens
caminho_original = "imagens/original"
caminho_original_cinza = "imagens/original_cinza"
caminho_preprocessamento = "imagens/filtragem"
caminho_segmentadas = "imagens/segmentadas"
caminho_histogramas = "imagens/histogramas"

# Criar diretórios de saída, se não existirem
os.makedirs(caminho_original_cinza, exist_ok=True)
os.makedirs(caminho_preprocessamento, exist_ok=True)
os.makedirs(caminho_segmentadas, exist_ok=True)
os.makedirs(caminho_histogramas, exist_ok=True)


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
def segmentar_com_otsu(image, nivel_branco=255, ajuste_limiar=True):
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

        # Calcula o histograma da imagem usando o NumPy
        hist, bins = np.histogram(cinza, bins=256, range=(0, 256))

        # Configuração do tamanho da figura
        fig = plt.figure(figsize=(8, 6))  # Ajuste o tamanho conforme necessário (largura, altura)
        plt.bar(bins[:-1], hist, width=bins[1]-bins[0], ec="black")
        plt.fill_between(bins[:-1], hist)

        # Configurações do plot
        plt.autoscale(enable=True, axis='both', tight=True)
        ax = fig.gca()
        ax.set_xticks(np.arange(0, 257, 32))
        ax.set_xticks(np.arange(0, 257, 16), minor=True)
        ax.set_yticks(np.arange(0, hist.max(), hist.max()//8), minor=False)
        ax.set_yticks(np.arange(0, hist.max(), hist.max()//4), minor=True)
        ax.grid(which='major', alpha=1.0)
        ax.grid(which='minor', alpha=0.5)
        ax.set_ylim(0, hist.max())
        ax.set_xlabel('Intensidades da imagem', fontsize='small')
        ax.set_ylabel('Histograma', fontsize='small')

        # Salvar o histograma na pasta 'histogramas'
        nome_histograma = os.path.splitext(arquivo)[0] + "_histograma_original_cinza.png"
        caminho_histograma = os.path.join(caminho_histogramas, nome_histograma)
        plt.savefig(caminho_histograma, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Fechar a figura para economizar memória

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
        imagem_segmentada = segmentar_com_otsu(imagem_preprocessada, ajuste_limiar=True)
        caminho_saida_segmentada = os.path.join(caminho_segmentadas, arquivo)
        cv2.imwrite(caminho_saida_segmentada, imagem_segmentada)

print("Filtragem e segmentacao concluidas")
