import os
import json
import psycopg2
import requests
from psycopg2.extras import execute_values
import numpy as np
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from typing import List, Dict, Any
from docling import document_converter

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações a partir das variáveis de ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/embeddings")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
OVERLAP = int(os.getenv("OVERLAP", "200"))

# Conexão com o PostgreSQL a partir das variáveis de ambiente
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "5432")
}

class EmbeddingGenerator:
    def __init__(self):
        self.setup_database()
        self.tables_created = set()

    def setup_database(self):
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Verifica se a extensão pgvector está disponível
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')")
        pgvector_available = cursor.fetchone()[0]

        if pgvector_available:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        conn.commit()
        cursor.close()
        conn.close()
        print("Banco de dados configurado com sucesso!")

    def create_document_table(self, filename):
        # Cria um nome de tabela seguro a partir do nome do arquivo
        table_name = f"documents_{self._sanitize_table_name(filename)}"
        
        # Verifica se a tabela já foi criada nesta sessão
        if table_name in self.tables_created:
            return table_name
            
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Verifica se a extensão pgvector está disponível
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')")
        pgvector_available = cursor.fetchone()[0]
        
        try:
            if pgvector_available:
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(3072),
                    chunk_index INTEGER NOT NULL,
                    section_title TEXT,
                    metadata JSONB
                )
                """)
            else:
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding FLOAT[],
                    chunk_index INTEGER NOT NULL,
                    section_title TEXT,
                    metadata JSONB
                )
                """)
                
            conn.commit()
            self.tables_created.add(table_name)
            print(f"Tabela {table_name} criada com sucesso!")
        except Exception as e:
            print(f"Erro ao criar tabela {table_name}: {e}")
        finally:
            cursor.close()
            conn.close()
            
        return table_name
    
    def _sanitize_table_name(self, filename):
        # Remove a extensão e substitui caracteres não permitidos por underscores
        base_name = os.path.splitext(filename)[0]
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', base_name)
        # Garante que o nome começa com uma letra
        if not sanitized[0].isalpha():
            sanitized = 'doc_' + sanitized
        # Limita o tamanho para evitar nomes muito longos
        return sanitized[:63]

    def chunk_markdown_by_headers(self, text):
        """Divide o texto markdown em chunks baseados em cabeçalhos que começam com ##"""
        # Padrão para encontrar cabeçalhos de nível 2 (##)
        header_pattern = r'(^|\n)##\s+.*?(?=\n##\s+|\Z)'
        
        # Verifica se há cabeçalhos de nível 2 no texto
        if not re.search(r'(\n|^)##\s+', text):
            # Se não houver cabeçalhos, usa o método padrão de chunking
            return self.chunk_text(text), []
        
        # Encontra todos os chunks que começam com um cabeçalho de nível 2
        chunks = []
        titles = []
        matches = re.finditer(header_pattern, text, re.DOTALL)
        
        for match in matches:
            chunk = match.group(0)
            # Extrai o título do cabeçalho (##)
            title_match = re.match(r'(?:^|\n)##\s+(.*?)(?:\n|$)', chunk)
            title = title_match.group(1).strip() if title_match else ""
            chunks.append(chunk)
            titles.append(title)
        
        return chunks, titles

    def chunk_text(self, text, chunk_size=None, overlap=None):
        if chunk_size is None:
            chunk_size = CHUNK_SIZE
        if overlap is None:
            overlap = OVERLAP
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) >= chunk_size / 2:
                chunks.append(chunk)
        return chunks

    def create_embedding(self, text):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        data = {
            "input": text,
            "model": EMBEDDING_MODEL
        }
        response = requests.post(OPENAI_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            print(f"Erro ao criar embedding: {response.text}")
            return None

    def extract_text_from_pdf(self, filepath):
        try:
            markdown = document_converter(filepath)
            return markdown
        except Exception as e:
            print(f"Erro ao converter {filepath} para Markdown com docling: {e}")
            return ""

    def process_file(self, filepath):
        filename = os.path.basename(filepath)
        text = ""

        if filename.endswith('.pdf'):
            text = self.extract_text_from_pdf(filepath)
        else:
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='latin-1') as file:
                    text = file.read()

        if not text.strip():
            print(f"Nenhum texto extraído de {filename}. Pulando arquivo.")
            return []

        # Criar uma tabela específica para este documento
        table_name = self.create_document_table(filename)
        
        # Chunk o texto baseado em cabeçalhos ##
        chunks, titles = self.chunk_markdown_by_headers(text)
        if not titles:  # Se não houver títulos (não usou chunking por cabeçalho)
            chunks = self.chunk_text(text)
            titles = [""] * len(chunks)  # Sem títulos para chunks regulares
            
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        document_ids = []

        for i, (chunk, title) in enumerate(zip(chunks, titles)):
            embedding = self.create_embedding(chunk)
            if not embedding:
                continue

            cursor.execute(
                f"INSERT INTO {table_name} (filename, content, embedding, chunk_index, section_title, metadata) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (filename, chunk, embedding, i, title, json.dumps({"source": filepath}))
            )
            document_id = cursor.fetchone()[0]
            document_ids.append(document_id)

        conn.commit()
        cursor.close()
        conn.close()
        return document_ids

    def process_directory(self, directory_path):
        processed_files = []
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            if os.path.isfile(filepath) and filename.endswith(('.txt', '.md', '.json', '.csv', '.pdf')):
                print(f"Processando arquivo: {filename}")
                document_ids = self.process_file(filepath)
                if document_ids:
                    processed_files.append({
                        "filename": filename,
                        "document_ids": document_ids
                    })
                    print(f"Arquivo {filename} processado com sucesso!")
                else:
                    print(f"Arquivo {filename} não pôde ser processado.")
        return processed_files

class EmbeddingCrew:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.embedding_generator = EmbeddingGenerator()

    def setup_crew(self):
        embedding_agent = Agent(
            role="Embedding Generator",
            goal="Gerar embeddings de alta qualidade para documentos de texto",
            backstory="Especialista em NLP e criação de embeddings",
            verbose=True,
            allow_delegation=False
        )
        database_agent = Agent(
            role="Database Manager",
            goal="Gerenciar armazenamento de embeddings no banco PostgreSQL",
            backstory="Especialista em bancos de dados e pgvector",
            verbose=True,
            allow_delegation=False
        )
        setup_task = Task(
            description="Configurar o banco de dados para armazenar documentos e embeddings",
            agent=database_agent,
            expected_output="Banco configurado"
        )
        process_task = Task(
            description=f"Processar arquivos de texto do diretório {self.directory_path} e gerar embeddings",
            agent=embedding_agent,
            expected_output="Arquivos processados e embeddings criados",
            context=[setup_task]
        )
        crew = Crew(
            agents=[embedding_agent, database_agent],
            tasks=[setup_task, process_task],
            verbose=True
        )
        return crew

    def run(self):
        if not os.path.isdir(self.directory_path):
            print(f"O diretório {self.directory_path} não existe.")
            return False
        files_to_process = [f for f in os.listdir(self.directory_path)
                            if os.path.isfile(os.path.join(self.directory_path, f))
                            and f.endswith(('.txt', '.md', '.json', '.csv', '.pdf'))]
        if not files_to_process:
            print(f"Aviso: Nenhum arquivo compatível encontrado no diretório {self.directory_path}")
            return []
        processed_files = self.embedding_generator.process_directory(self.directory_path)
        if processed_files:
            print(f"Total de arquivos processados: {len(processed_files)}")
            print("Embeddings gerados e armazenados com sucesso!")
        else:
            print("Nenhum arquivo foi processado com sucesso.")
        return processed_files

def validate_env():
    required_vars = [
        "OPENAI_API_KEY",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST"
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"Erro: Faltando variáveis de ambiente: {', '.join(missing)}")
        return False
    return True

if __name__ == "__main__":
    import sys
    if not validate_env():
        sys.exit(1)
    directory_path = os.getenv("DOCUMENTS_DIR", "data/documents")
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Diretório {directory_path} criado com sucesso.")
        except Exception as e:
            print(f"Erro ao criar diretório: {e}")
            sys.exit(1)

    embedding_crew = EmbeddingCrew(directory_path)
    result = embedding_crew.run()
    if result:
        print("Processo de geração de embeddings concluído com sucesso!")
        sys.exit(0)
    else:
        print("Nenhum arquivo foi processado.")
        sys.exit(0)