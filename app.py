import os

import sys

from langchain_ollama import OllamaLLM

def load_docs():
    
    try:
        with open("company_rules.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print("Ошибка: Сначала создай файл company_rules.txt с текстом!")
        sys.exit(1)

def main():
    print("--- Локальный ИИ-помощник запущен ---")
    
    
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    
    llm = OllamaLLM(base_url=ollama_url, model="qwen2.5:1.5b", temperature=0.2)
    context = load_docs()
    
    print("База знаний загружена. Спрашивай что хочешь (для выхода набери 'exit').")
    
    while True:
        try:
            query = input("\nВопрос > ").strip()
        except (KeyboardInterrupt, EOFError):
            
            print("\nПока!")
            break
            
        if not query:
            continue
            
        if query.lower() in ['exit', 'quit', 'выход']:
            print("Выходим...")
            break
            
        # (Если не умеете писать пронт можно забить через ИИ) иногда Qwen выдает ответы с иероглифами так что строчка -" Отвечай строго на русском языке обязательно нужна"
        prompt = f"""Контекст:
{context}

Задание: Ответь на вопрос пользователя, используя только текст выше. 
Если в тексте ответа нет, напиши просто: "В базе знаний нет такой информации".
Отвечай строго на русском языке.

Вопрос: {query}
Ответ:"""
        
        try:
            res = llm.invoke(prompt)
            print(f"\nОтвет:\n{res.strip()}")
        except Exception as e:
            print(f"\nЧто-то пошло не так: {e}")

if __name__ == "__main__":
    main()