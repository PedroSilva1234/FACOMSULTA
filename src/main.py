import os
import sys
from dotenv import load_dotenv

from src.RAG.gerar_respostas import iniciar_FacomSulta

#1 Certifique-se de que o diretório atual está no sys.path para que os módulos possam ser encontrados
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. Carrega as variáveis de ambiente do .env ANTES de carregar a IA
load_dotenv()

def main():
    print("Iniciando o FACOMSULTA...")
    # Carrega as variáveis de ambiente do arquivo .env

    try: 
        iniciar_FacomSulta()

    except KeyboardInterrupt:
        print("\nFACOMSULTA interrompido pelo usuário.")
        sys.exit(0)
    except Exception as e: 
        print(f"Ocorreu um erro ao iniciar o FACOMSULTA: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()