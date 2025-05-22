from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from typing import Dict, List, Optional, Any
import os
import psycopg2
import requests
import numpy as np
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configurações de banco de dados
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "5432")
}

# API OpenAI para embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/embeddings")
OPENAI_COMPLETION_URL = os.getenv("OPENAI_COMPLETION_URL", "https://api.openai.com/v1/chat/completions")
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL", "gpt-4o-mini")

class EmbeddingSearchTool(BaseTool):
    name: str = "Embedding Search Tool"
    description: str = "Realiza buscas semânticas em um banco de dados usando embeddings."

    def __init__(self):
        super().__init__()

    def _get_db_connection(self):
        try:
            connection = psycopg2.connect(**DB_CONFIG)
            return connection
        except Exception as e:
            print(f"Erro ao conectar ao banco de dados: {e}")
            return None

    def _get_all_tables(self):
        connection = self._get_db_connection()
        if not connection:
            return []
        
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            connection.close()
            return tables
        except Exception as e:
            if connection:
                connection.close()
            print(f"Erro ao listar tabelas: {str(e)}")
            return []

    def _run(self, query: str, table_name: str = None, limit: int = 1) -> str:
        connection = self._get_db_connection()
        if not connection:
            return "Erro: Não foi possível estabelecer conexão com o banco de dados."

        try:
            query_embedding = self.create_embedding(query)
            if not query_embedding:
                connection.close()
                return "Erro: Não foi possível gerar o embedding para a consulta."

            # Se nenhuma tabela específica foi fornecida, buscar em todas as tabelas
            if not table_name:
                tables = self._get_all_tables()
                best_result = None
                best_table = None
                
                for table in tables:
                    # Verifica se a tabela tem as colunas necessárias
                    cursor = connection.cursor()
                    cursor.execute(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = %s AND column_name IN ('embedding', 'content')
                    """, (table,))
                    columns = cursor.fetchall()
                    cursor.close()
                    
                    if len(columns) < 2:  # Precisa ter embedding e content
                        continue
                    
                    result = self._search_in_table(connection, table, query_embedding, limit)
                    if result and (not best_result or (isinstance(result, dict) and result.get('score', 0) > best_result.get('score', 0))):
                        best_result = result
                        best_table = table
                
                if best_result and best_table:
                    response = self.generate_response(query, best_result['content'])
                    connection.close()
                    return f"Documento consultado: {best_table}\nConteúdo da resposta: {response}"
                else:
                    connection.close()
                    return "Nenhum resultado relevante encontrado em nenhuma tabela."
            else:
                # Busca apenas na tabela especificada
                result = self._search_in_table(connection, table_name, query_embedding, limit)
                connection.close()
                
                if result:
                    response = self.generate_response(query, result['content'])
                    return f"Documento consultado: {table_name}\nConteúdo da resposta: {response}"
                else:
                    return f"Nenhum resultado encontrado para '{query}' na tabela '{table_name}'."

        except Exception as e:
            if connection:
                connection.close()
            return f"Erro ao realizar busca: {str(e)}"

    def _search_in_table(self, connection, table_name, query_embedding, limit=1):
        try:
            cursor = connection.cursor()
            
            cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            pgvector_available = cursor.fetchone()[0]

            cursor.execute(f"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (table_name,))
            table_exists = cursor.fetchone()[0]

            if not table_exists:
                cursor.close()
                return None

            if pgvector_available:
                cursor.execute(f"""
                    SELECT content, (embedding <=> %s::vector) as distance
                    FROM {table_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (str(query_embedding), str(query_embedding), limit))
            else:
                cursor.execute(f"""
                    SELECT content, (embedding <#> %s::vector) as distance
                    FROM {table_name}
                    ORDER BY embedding <#> %s::vector
                    LIMIT %s
                """, (str(query_embedding), str(query_embedding), limit))

            results = cursor.fetchall()
            cursor.close()
            
            if not results:
                return None
                
            content = results[0][0]
            distance = results[0][1]
            # Calcular um score simples (quanto menor a distância, maior o score)
            score = 1.0 - min(1.0, max(0.0, distance))
            
            return {"content": content, "score": score}
            
        except Exception as e:
            print(f"Erro ao buscar na tabela {table_name}: {str(e)}")
            return None

    def create_embedding(self, text: str) -> List[float]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        data = {
            "input": text,
            "model": EMBEDDING_MODEL
        }
        try:
            response = requests.post(OPENAI_API_URL, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()["data"][0]["embedding"]
            else:
                print(f"Erro ao criar embedding: {response.text}")
                return None
        except Exception as e:
            print(f"Exceção ao criar embedding: {e}")
            return None

    def generate_response(self, question: str, context: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        data = {
            "model": COMPLETION_MODEL,
            "messages": [
                {"role": "system", "content": "Você é um assistente que responde com base em documentos internos."},
                {"role": "user", "content": f"Com base no seguinte trecho de documento, responda a pergunta de forma clara e objetiva.\n\nTrecho: {context}\n\nPergunta: {question}"}
            ]
        }
        try:
            response = requests.post(OPENAI_COMPLETION_URL, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Erro na geração de resposta: {response.text}"
        except Exception as e:
            return f"Exceção na geração de resposta: {e}"


class EmbeddingSearchAgent:
    def __init__(self):
        self.search_tool = EmbeddingSearchTool()

    def create_agent(self):
        search_agent = Agent(
            role="Especialista em Busca Semântica",
            goal="Fornecer respostas claras e objetivas com base no documento mais relevante.",
            backstory=(
                "Sou especialista em buscas por similaridade semântica, capaz de encontrar o trecho mais relevante e gerar uma resposta bem formulada."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[self.search_tool]
        )
        return search_agent

    def create_search_task(self, query: str, table_name: str = None):
        agent = self.create_agent()
        task = Task(
            description=f"Buscar a melhor resposta para: '{query}'" + (f" usando documentos em '{table_name}'" if table_name else ""),
            agent=agent,
            expected_output="Uma resposta baseada no trecho mais relevante do documento"
        )
        return task

    def setup_crew(self):
        agent = self.create_agent()
        crew = Crew(
            agents=[agent],
            tasks=[],
            verbose=True
        )
        return crew

    def run_search(self, query: str, table_name: str = None):
        return self.search_tool._run(query, table_name)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Erro: OPENAI_API_KEY não encontrada nas variáveis de ambiente.")
        exit(1)

    if not all([os.getenv(key) for key in ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST"]]):
        print("Erro: Configurações de banco de dados incompletas nas variáveis de ambiente.")
        exit(1)

    search_agent_manager = EmbeddingSearchAgent()
    
    # Solicita a consulta pelo terminal
    query = input("Digite sua pergunta: ")
    # Opcionalmente, pode-se especificar uma tabela
    table_name = input("Digite o nome da tabela (ou deixe em branco para buscar em todas): ").strip() or None
    
    print(f"Realizando busca para: '{query}'")
    results = search_agent_manager.run_search(query, table_name)
    print(results)