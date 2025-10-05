from prefect import task, flow
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import time
import json
import os
from datetime import datetime


@task
def extract_files_publication(csv_path, max_retries=3, delay=3, batch_size=10, cache_file='docs_cache.json'):
    """
    Carrega documentos das URLs com retry, cache e processamento em lotes
    """
    print(f"📂 Iniciando extração de arquivos do CSV: {csv_path}")

    # Carregar CSV
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"✅ CSV carregado com sucesso: {len(df)} linhas encontradas")
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        raise

    links = df['Link'].tolist()
    print(f"🔗 Total de links extraídos: {len(links)}")
    print(f"📋 Primeiros 3 links: {links[:3]}")

    # Carregar cache se existir
    cached_docs = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                cached_docs = {item['url']: item for item in cached_data}
            print(f"💾 Cache carregado: {len(cached_docs)} documentos já baixados")
        except Exception as e:
            print(f"⚠️ Erro ao carregar cache: {e}")

    docs = []
    failed_links = []
    skipped = 0

    print(f"\n{'=' * 70}")
    print(f"🚀 Iniciando download (processando {batch_size} por vez)")
    print(f"{'=' * 70}\n")

    for i, link in enumerate(links, 1):
        # Verificar se já está no cache
        if link in cached_docs:
            doc = Document(
                page_content=cached_docs[link]['content'],
                metadata=cached_docs[link]['metadata']
            )
            docs.append(doc)
            skipped += 1
            if i % 50 == 0:
                print(f"[{i}/{len(links)}] ⚡ {skipped} docs do cache | {len(docs)} total")
            continue

        success = False

        for attempt in range(max_retries):
            try:
                print(f"[{i}/{len(links)}] 🌐 Baixando: {link[:60]}... (tentativa {attempt + 1}/{max_retries})")

                # Configurar timeout e headers para evitar bloqueio
                loader = WebBaseLoader([link])
                loader.requests_per_second = 0.3  # Mais devagar
                loader.requests_kwargs = {
                    'timeout': 30,
                    'headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                }

                doc = loader.load()

                if doc:
                    docs.extend(doc)

                    # Salvar no cache
                    cached_docs[link] = {
                        'url': link,
                        'content': doc[0].page_content,
                        'metadata': doc[0].metadata,
                        'downloaded_at': datetime.now().isoformat()
                    }

                    print(f"    ✅ Sucesso! ({len(doc[0].page_content)} chars)")
                    success = True
                    break
                else:
                    print(f"    ⚠️ Documento vazio")

            except Exception as e:
                error_msg = str(e)[:100]
                print(f"    ❌ Erro: {error_msg}")

                if attempt < max_retries - 1:
                    wait_time = delay * (attempt + 2)
                    print(f"    ⏳ Aguardando {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    failed_links.append({'url': link, 'error': error_msg})

        if not success:
            print(f"    💥 Falha definitiva após {max_retries} tentativas")

        # Salvar cache a cada 10 documentos
        if i % 10 == 0:
            _save_cache(cached_docs, cache_file)
            print(f"    💾 Cache salvo ({len(cached_docs)} docs)")

        # Delay entre requisições
        if success and i < len(links):
            time.sleep(delay)

        # Relatório de progresso
        if i % 50 == 0 or i == len(links):
            success_count = len(docs)
            fail_count = len(failed_links)
            progress = (i / len(links)) * 100
            print(f"\n{'─' * 70}")
            print(f"📊 PROGRESSO: {progress:.1f}% ({i}/{len(links)})")
            print(f"   ✅ Sucessos: {success_count} | ❌ Falhas: {fail_count} | ⚡ Cache: {skipped}")
            print(f"{'─' * 70}\n")

    # Salvar cache final
    _save_cache(cached_docs, cache_file)

    # Salvar falhas
    if failed_links:
        _save_failed_links(failed_links)

    print(f"\n{'=' * 70}")
    print(f"📊 RESUMO FINAL:")
    print(f"   ✅ Documentos carregados: {len(docs)}")
    print(f"   ❌ Falhas: {len(failed_links)}")
    print(f"   ⚡ Do cache: {skipped}")
    print(f"   💾 Cache total: {len(cached_docs)}")
    print(f"{'=' * 70}\n")

    return docs, failed_links


def _save_cache(cached_docs, cache_file):
    """Salva cache em arquivo JSON"""
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(list(cached_docs.values()), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Erro ao salvar cache: {e}")


def _save_failed_links(failed_links, output_file='failed_links.json'):
    """Salva links que falharam"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(failed_links, f, ensure_ascii=False, indent=2)
        print(f"📄 Links com falha salvos em: {output_file}")
    except Exception as e:
        print(f"⚠️ Erro ao salvar falhas: {e}")


@task
def split_documents(docs, chunk_size=1000, chunk_overlap=100):
    """Divide documentos em chunks menores"""
    if not docs:
        print("⚠️ Nenhum documento para dividir")
        return []

    print(f"\n📝 Dividindo {len(docs)} documentos em chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = text_splitter.split_documents(docs)

    print(f"✅ {len(docs)} documentos → {len(split_docs)} chunks")
    print(f"📊 Média de {len(split_docs) / len(docs):.1f} chunks por documento")

    return split_docs


@task
def generate_embeddings_vectorstore(split_docs):
    """Cria um vector store in-memory usando FAISS"""
    if not split_docs:
        print("⚠️ Nenhum documento para criar embeddings")
        return None

    try:
        print(f"\n🔄 Gerando embeddings para {len(split_docs)} chunks...")
        print(f"⏳ Isso pode levar alguns minutos...")

        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
            documents=split_docs,
            embedding=embeddings_model
        )

        print(f"✅ Vectorstore in-memory criado com sucesso!")
        return vectorstore

    except Exception as e:
        print(f"❌ Erro ao criar vectorstore: {e}")
        return None


@task
def save_vectorstore(vectorstore, save_path='./vectorstore'):
    """Salva o vectorstore localmente"""
    if vectorstore:
        try:
            print(f"\n💾 Salvando vectorstore em: {save_path}")
            vectorstore.save_local(save_path)
            print(f"✅ Vectorstore salvo com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao salvar vectorstore: {e}")


@flow
def pipeline_publication(
        csv_path='../data/SB_publication_PMC.csv',
        save_local=True,
        max_retries=3,
        delay=3,
        batch_size=10
):
    """
    Pipeline completo de processamento de publicações

    Args:
        csv_path: Caminho do CSV com os links
        save_local: Se deve salvar o vectorstore localmente
        max_retries: Número de tentativas por URL
        delay: Delay em segundos entre requisições
        batch_size: Tamanho do lote (não usado atualmente)
    """
    print(f"\n{'#' * 70}")
    print(f"# 🚀 PIPELINE DE PROCESSAMENTO DE PUBLICAÇÕES")
    print(f"{'#' * 70}\n")

    start_time = time.time()

    # 1. Extrai documentos das URLs
    docs, failed_links = extract_files_publication(
        csv_path,
        max_retries=max_retries,
        delay=delay,
        batch_size=batch_size
    )

    # 2. Verifica se conseguiu carregar documentos
    if not docs:
        print("❌ Nenhum documento foi carregado com sucesso")
        return None

    # 3. Divide documentos
    split_docs = split_documents(docs)

    # 4. Cria vectorstore
    vectorstore = generate_embeddings_vectorstore(split_docs)

    # 5. Salva localmente se solicitado
    if save_local and vectorstore:
        save_vectorstore(vectorstore)

    # Tempo total
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print(f"\n{'#' * 70}")
    print(f"# ✅ PIPELINE CONCLUÍDO!")
    print(f"# ⏱️  Tempo total: {minutes}min {seconds}s")
    print(f"# 📄 {len(docs)} documentos → {len(split_docs)} chunks")
    print(f"# ❌ {len(failed_links)} falhas")
    print(f"{'#' * 70}\n")

    return vectorstore


if __name__ == "__main__":
    # Executa o pipeline
    vs = pipeline_publication(
        csv_path='../data/SB_publication_PMC.csv',
        save_local=True,
        max_retries=3,
        delay=3  # 3 segundos entre cada requisição
    )

    # Testa o vectorstore
    if vs:
        print("\n" + "=" * 70)
        print("🔍 TESTANDO BUSCA NO VECTORSTORE")
        print("=" * 70 + "\n")

        test_queries = [
            "space biology",
            "microgravity effects",
            "radiation exposure"
        ]

        for query in test_queries:
            print(f"\n🔎 Query: '{query}'")
            results = vs.similarity_search(query, k=2)

            for i, doc in enumerate(results, 1):
                print(f"\n  📄 Resultado {i}:")
                print(f"     {doc.page_content[:200]}...")
                print(f"     Fonte: {doc.metadata.get('source', 'N/A')}")