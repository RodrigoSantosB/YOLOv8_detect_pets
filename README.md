# IMPORTANTE QUE SEJA RODADO EM GPU (se for na máquina local)

0. Baixar e instalar drivers Nvdia para sua versão de placa:
    `https://www.edivaldobrito.com.br/driver-nvidia-no-linux/`    

1. Baixar e configurar CUDA (11.x ou 12.x) na sua maquina:
    * isso pode ser feito por meio da documentação (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), tenha cuidado com a versão do ubuntu que está  usando. A instalação costuma ser muito expecífica (versões que fucionam bem 18.04, 20.04, 22.04 LTS)

2. Instalar o pytorch via pip, baixar o ultralytics via pip:
   ```
    pip install torch
    pip install ultralytics
   ```
4. Treinar o modelo, instanciando o modelo na linha
   ``` `model = YOLO("yolov8n.yaml")` ```
   e configurando corretamente o path de onde seus arquivos estao ex:
    ``` `path: /home/usr/desktop/pasta/visao/pasta_que contem_os_arquvos_yolov8` ```
