# Instruções para Rodar o Código de Filtragem e Segmentação de Imagens

## 1. Pré-requisitos
Antes de executar o código, certifique-se de que todos os requisitos estão instalados no seu ambiente.

### Dependências:
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- scikit-image

### Instalação:
Instale as bibliotecas necessárias usando o seguinte comando:
```bash
pip install opencv-python numpy scikit-image
```

---

## 2. Estrutura de Pastas
Organize as imagens e os diretórios conforme a estrutura abaixo:

```
- imagens/
  - original/               # Imagens originais para processamento
  - original_cinza/         # Imagens convertidas para escala de cinza
  - filtragem/              # Imagens filtradas (passa-baixa + ruído gaussiano)
  - segmentadas/            # Imagens segmentadas (utilizando Otsu)
- script.py                 # Código Python para processamento e segmentação
```

Certifique-se de que as pastas **original**, **original_cinza**, **filtragem** e **segmentadas** existam antes de rodar o código.

---

## 3. Executando o Código
Para executar o script, siga os passos abaixo:

1. Salve o código em um arquivo Python, por exemplo, `processar_imagens.py`.
2. Coloque as imagens originais na pasta `imagens/original`.
3. Execute o script com o comando:
   ```bash
   python processar_imagens.py
   ```

---

## 4. O que o Código Faz

1. **Conversão para Escala de Cinza**:  
   As imagens na pasta `original` serão convertidas para escala de cinza e salvas na pasta `original_cinza`.

2. **Filtragem de Passa-Baixa e Ruido Gaussiano**:  
   As imagens em escala de cinza serão suavizadas para reduzir ruídos, e os resultados serão salvos na pasta `filtragem`.

3. **Segmentação com Otsu**:  
   As imagens filtradas serão segmentadas para destacar áreas de interesse, com a saída salva na pasta `segmentadas`.

---

## 5. Resultados Esperados

- **original_cinza/**: Imagens convertidas para escala de cinza.
- **filtragem/**: Imagens suavizadas com filtro de passa-baixa e ruido gaussiano.
- **segmentadas/**: Imagens binárias com regiões de interesse destacadas após segmentação.

---

## 6. Personalização

- **Ajuste do Limiar Manual**:  
  Caso queira ajustar o limiar manualmente durante a segmentação, modifique o valor de `limiar_manual` no script:
  ```python
  limiar_manual = 100  # Ajuste conforme necessário
  ```

- **Nível de Branco**:  
  Para alterar o nível de branco nas áreas segmentadas, ajuste o parâmetro `nivel_branco` ao chamar a função `segmentar_com_otsu`.

---

## 7. Observações Finais

- Certifique-se de que os caminhos das pastas estão corretos.
- Verifique se as imagens no formato `.png` estão corretamente salvas na pasta `original`.
- O método de segmentação utilizado (Otsu) pode ser ajustado para diferentes tipos de imagens e objetivos específicos.

---
