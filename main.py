from importlib.metadata import metadata
import pandas as pd
import numpy as np
from openai import OpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import ast
from langchain_core.embeddings import Embeddings
from typing import List
import time


# Функция загрузки датасета
def load_embeddings_to_dataframe(filepath):
    """
    Загружает данные эмбеддингов из CSV файла в pandas DataFrame

    Args:
        filepath (str): Путь к файлу с данными

    Returns:
        pd.DataFrame: DataFrame с загруженными данными
    """
    try:
        df0 = pd.read_csv(filepath)
        print(f"Загружено {len(df0)} записей из файла {filepath}")
        print(f"Колонки: {df0.columns.tolist()}")
        print(f"\nПервые 3 записи:")
        print(df0.head(3))
        return df0
    except Exception as e:
        print(f"Произошла ошибка при загрузке данных: {e}")
        return None

# Загрузка данных
df = load_embeddings_to_dataframe("arxiv_embeddings202505211515.csv")

# Просмотр информации о датасете
print(f"Размер датасета: {df.shape}")
print(f"\nТипы данных:\n{df.dtypes}")
print(f"\nПроверка пропущенных значений:\n{df.isnull().sum()}")

# Статистика по категириям статей
static = df['main_category'].value_counts()
print("Статистика по категориям\n", static.head(10))

df = df.head(1000)
print('Подмножество из ',len(df),' статей')

#Подготовка документов в формате LangChain
def prepare_documents(df, limit=1000):
    """
    Преобразует DataFrame в список документов LangChain

    Args:
        df: DataFrame с данными статей
        limit: Количество документов для обработки

    Returns:
        List[Document]: Список документов
    """
    documents = []

    # Возьмите первые limit записей
    df_subset = df.head(limit)

    for idx, row in df_subset.iterrows():
        # Формируйте текст документа из доступных полей
        page_content = ''
        # Адаптируйте под структуру вашего датасета
        if 'title' in df.columns and pd.notna(row['title']):
            page_content += f"Название: {row['title']}\n"
        if 'abstract' in df.columns and pd.notna(row['abstract']):
            page_content += f"Аннотация: {row['abstract']}\n"
        if 'authors' in df.columns and pd.notna(row['authors']):
            page_content += f"Авторы: {row['authors']}\n"

        # Метаданные
        metadata = {}
        if 'categories' in df.columns and pd.notna(row['categories']):
            metadata['categories'] = row['categories']
        if 'year' in df.columns and pd.notna(row['year']):
            metadata['year'] = row['year']
        if 'article_id' in df.columns and pd.notna(row['article_id']):
            metadata['article_id'] = row['article_id']

        documents.append(Document(
            page_content=page_content.strip(),
            metadata=metadata
        ))

    print(f"Подготовлено {len(documents)} документов\n")
    return documents

# Подготовка документов
documents = prepare_documents(df, limit=1000)

# Просмотр примера документа
print("Пример документа:\n")
print(f"Содержимое: {documents[0].page_content[:200]}...\n")
print(f"Метаданные: {documents[0].metadata}\n")


client = OpenAI(
api_key="ZmJhMjUwZTItMDg0ZC00N2E3LWIyNDktYjA4MTQyZGFmMGE4.97f6d089a16317c3aa93b365eda739a8",
    base_url="https://foundation-models.api.cloud.ru/v1"
)


def get_embedding(text: str, model="BAAI/bge-m3") -> list:
    """Получает эмбеддинг текста"""
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Тестирование функции эмбеддингов
test_embedding = get_embedding("Тестовый запрос")
print(f"Размерность эмбеддинга: {len(test_embedding)}")


class CustomEmbeddings(Embeddings):
    """Кастомный класс эмбеддингов для работы с API"""

    def __init__(self, client, model="BAAI/bge-m3"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получение эмбеддингов для списка документов"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Получение эмбеддинга для запроса"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding


# Создание экземпляра эмбеддингов
embeddings = CustomEmbeddings(client)


# Создание векторного хранилища ChromaDB
print("Создание векторного хранилища...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="arxiv_papers",
    persist_directory="./chroma_db"  # Директория для сохранения
)
print("Векторное хранилище успешно создано!")




# Проверка работы хранилища
test_query = "машинное обучение и нейронные сети"
results = vectorstore.similarity_search(test_query, k=3)
print(f"\nРезультаты поиска по запросу '{test_query}':")
for i, doc in enumerate(results, 1):
    print(f"\n{i}. {doc.page_content[:200]}...")
    print(f"   Метаданные: {doc.metadata}\n")



# Разные k
test_queries = ["машинное обучение", "нейронные сети", "компьютерное зрение"]
for query in test_queries:
    print(f"\n=== Запрос: '{query}' ===")

    for k in [1, 3, 5, 10]:  # Разные значения k
        print(f"\nk = {k}:")
        results = vectorstore.similarity_search(query, k=k)

        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content[:150]}...")

# Разные типы запросов
thematic_queries = [
    "трансформер архитектура в NLP",
    "обучение с подкреплением для робототехники",
    "глубокое обучение",
    "SVM классификация",
]

for query in thematic_queries:
    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results, 1):
        print(f"Содержимое: {doc.page_content[:200]}...")
        print(f"Метаданные: {doc.metadata}")


# Получение оценок схожести
score_queries = [
    "компьютерное зрение и распознавание объектов",
    "обработка естественного языка",
    "генеративные модели"
]

for query in score_queries:
    print(f"\nЗапрос: '{query}'")

    results_with_scores = vectorstore.similarity_search_with_score(query, k=5)

    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\n{i}. Оценка: {score:.4f}")
        print(f"   Содержимое : {doc.page_content.split('Название: ')[1].split('\\n')[0][:120]}...")
        print(f"   Категория: {doc.metadata.get('categories', 'Нет')}")



# Создание ретривера с поиском по схожести
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Возвращать топ-5 документов
)

# Тестирование ретривера
query = "глубокое обучение для обработки изображений"
retrieved_docs = retriever.invoke(query)

print(f"Найдено документов: {len(retrieved_docs)}")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nДокумент {i}:")
    print(doc.page_content[:150] + "...")

# MMR балансирует между релевантностью и разнообразием результатов
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,  # Количество документов для первичной выборки
        "lambda_mult": 0.5  # Баланс между релевантностью (1.0) и разнообразием (0.0)
    }
)

# Сравнение результатов
query = "обработка естественного языка"
docs_similarity = retriever.invoke(query)
docs_mmr = retriever_mmr.invoke(query)

print("Сравнение результатов поиска:")
print("\n=== Similarity Search ===")
for i, doc in enumerate(docs_similarity[:3], 1):
    print(f"{i}. {doc.page_content[:100]}...")

print("\n=== MMR Search ===")
for i, doc in enumerate(docs_mmr[:3], 1):
    print(f"{i}. {doc.page_content[:100]}...")

# Ретривер с фильтрацией по оценке схожести
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,  # Минимальная оценка схожести
        "k": 10
    }
)

# ВАЖНО: Не все векторные хранилища поддерживают score_threshold
# В случае ChromaDB может потребоваться другой подход


# Различный lambda_mult в MMR
lambda_values = [0.0, 0.5, 1.0]
query = "глубокое обучение"

for lambda_val in lambda_values:
    print(f"\nlambda_mult = {lambda_val}:")

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": lambda_val
        }
    )
    docs = retriever.invoke(query)
    for i, doc in enumerate(docs, 1):
        title = doc.page_content.split("Название:")[1].split("\n")[0][
            :50] if "Название:" in doc.page_content else doc.page_content[:50]
        print(f"  {i}. {title}...")

# Сравнение времени работы
def compare_search_time(query, num_runs=3):
    search_configs = [
        ("similarity", {"k": 5}),
        ("mmr_lambda_0.0", {"k": 5, "fetch_k": 20, "lambda_mult": 0.0}),
        ("mmr_lambda_0.5", {"k": 5, "fetch_k": 20, "lambda_mult": 0.5}),
        ("mmr_lambda_1.0", {"k": 5, "fetch_k": 20, "lambda_mult": 1.0})
    ]
    results = {}
    for search_name, kwargs in search_configs:
        search_type = search_name.split("_")[0] if "_" in search_name else search_name

        retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=kwargs
        )
        times = []
        for _ in range(num_runs):
            start = time.time()
            retriever.invoke(query)
            end = time.time()
            times.append((end - start) * 1000)  # мс

        avg_time = sum(times) / len(times)
        results[search_name] = avg_time

        print(f"{search_name}: {avg_time:.1f} мс")

    return results


# Тестируем функцию
test_query = "нейронные сети"
print(f"\nЗапрос: '{test_query}'")
results = compare_search_time(test_query)



llm = GigaChat(
    credentials="MDE5OWM0YWYtOTM2My03YmU5LWJmNDQtMmU5ZDYxNDkwMDMzOmVlM2UwZWQyLWRjODUtNGUwMy1hZDIwLWM1N2VkY2U3MmY2Nw==",
    scope="GIGACHAT_API_B2B",
    model="Gigachat-2-pro",
    verify_ssl_certs=False,
    timeout=30
)

# Тест языковой модели
test_response = llm.invoke("Привет! Ответь кратко: что такое машинное обучение?")
print(f"Ответ модели: {test_response.content}")


# Шаблон промпта для RAG
prompt_template = """Ты -- научный ассистент, специализирующийся на анализе научных статей.
Твоя задача -- отвечать на вопросы пользователя, основываясь ТОЛЬКО на предоставленном контексте из научных статей ArXiv.

Правила:
1. Используй только информацию из контекста ниже
2. Если в контексте нет информации для ответа, честно скажи об этом
3. Указывай, из каких статей взята информация (если есть метаданные)
4. Отвечай на русском языке, четко и структурированно
5. Если вопрос касается технических деталей, будь точным

Контекст из научных статей:
{context}

Вопрос пользователя: {question}

Ответ:"""

prompt = ChatPromptTemplate.from_template(prompt_template)


def format_docs(docs):
    """
    Форматирует список документов в единую строку контекста
    Args:
        docs: Список документов Document

    Returns:
        str: Форматированный контекст
    """
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Документ {i}]")
        context_parts.append(doc.page_content)
        if doc.metadata:
            context_parts.append(f"Метаданные: {doc.metadata}")
        context_parts.append("")  # Пустая строка для разделения
    return "\n".join(context_parts)


# Тест форматирования
test_docs = retriever.invoke("нейронные сети")
formatted_context = format_docs(test_docs[:2])
print("Пример форматированного контекста:")
print(formatted_context[:500] + "...")


# Создание RAG-цепочки
rag_chain = (
    {
        "context": retriever | format_docs,  # Извлекаем и форматируем документы
        "question": RunnablePassthrough()     # Передаем вопрос как есть
    }
    | prompt      # Формируем промпт
    | llm         # Отправляем в языковую модель
)

# Тестирование RAG-системы
questions = [
    "Какие методы машинного обучения используются для обработки изображений?",
    "Расскажи о применении трансформеров в обработке естественного языка",
    "Какие существуют подходы к обучению нейронных сетей?"
]

print("=== Тестирование RAG-системы ===\n")
for i, question in enumerate(questions, 1):
    print(f"Вопрос {i}: {question}")
    print("-" * 80)
    try:
        response = rag_chain.invoke(question)
        print(f"Ответ: {response.content}\n")
    except Exception as e:
        print(f"Ошибка: {e}\n")


def interactive_rag_qa():
    """Интерактивная система вопросов-ответов"""
    print("=== Интерактивная RAG-система для научных статей ArXiv ===")
    print("Введите 'выход' для завершения\n")

    while True:
        question = input("Ваш вопрос: ").strip()

        if question.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break

        if not question:
            continue

        try:
            # Получаем релевантные документы
            docs = retriever.invoke(question)
            print(f"\n📚 Найдено релевантных документов: {len(docs)}")

            # Генерируем ответ
            response = rag_chain.invoke(question)
            print(f"\n🤖 Ответ:\n{response.content}\n")

            # Показываем источники
            show_sources = input("Показать источники? (да/нет): ").strip().lower()
            if show_sources in ['да', 'yes', 'y', 'д']:
                print("\n📖 Источники:")
                for i, doc in enumerate(docs[:3], 1):
                    print(f"\n{i}. {doc.page_content[:200]}...")
                    print(f"   Метаданные: {doc.metadata}")

            print("\n" + "=" * 80 + "\n")

        except Exception as e:
            print(f"❌ Ошибка: {e}\n")


# Дополнительные эксперименты с разными темами и статьями

# Разные тематические запросы
experiment_queries = [
    ("Компьютерное зрение", "Как работают сверточные нейронные сети?"),
    ("NLP", "Что такое трансформеры в обработке текста?"),
    ("Робототехника", "Применение обучения с подкреплением в роботах"),
    ("Биоинформатика", "Использование ML в анализе ДНК"),
    ("Физика", "Применение нейронных сетей в квантовой механике")
]

for category, query in experiment_queries:
    print(f"\n{category}: {query}")

    try:
        response = rag_chain.invoke(query)
        print(f"Ответ (первые 150 символов): {response.content[:150]}...")
    except Exception as e:
        print(f"Ошибка: {e}")

# Разное количество документов
test_query = "градиентный спуск в нейронных сетях"

for k in [1, 3, 5]:
    print(f"\nk = {k}:")

    custom_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    custom_chain = (
            {"context": custom_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm
    )

    response = custom_chain.invoke(test_query)
    print(f"Ответ (первые 100 символов): {response.content[:100]}...")

# Сравнение поисковых стратегий
query = "методы регуляризации"

# Similarity
retriever_sim = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chain_sim = ({"context": retriever_sim | format_docs, "question": RunnablePassthrough()} | prompt | llm)

# MMR
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)
chain_mmr = ({"context": retriever_mmr | format_docs, "question": RunnablePassthrough()} | prompt | llm)

print("Similarity:", chain_sim.invoke(query).content[:100] + "...")
print("MMR:", chain_mmr.invoke(query).content[:100] + "...")