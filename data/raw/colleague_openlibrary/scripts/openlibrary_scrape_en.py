# MiniMax code with deduplication lines
import requests
import csv
import time
import re

def normalize(s):
    """Нормализация строки для надёжного сравнения."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)            # схлопываем лишние пробелы
    s = re.sub(r"[^\w\s]", "", s)        # убираем пунктуацию
    return s

def scrape_creative_writing():
    print("=== ЗАПУСК СКРЕПЕРА (Open Library: creative writing) ===")

    url = "https://openlibrary.org/search.json"
    base_params = {
        "subject": "creative writing",
        "limit": 100,
    }

    all_books = []
    seen = set()                # сюда кладём (title_norm, author_norm)
    page = 1
    total_expected = None
    max_pages = 200
    duplicates_total = 0

    while page <= max_pages:
        params = {**base_params, "page": page}
        print(f"Запрос страницы {page}…")

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if total_expected is None:
                total_expected = data.get("numFound", 0)
                print(f"Всего найдено по numFound: {total_expected}")

            docs = data.get("docs", [])
            if not docs:
                print("Страница пуста — выходим.")
                break

            new_in_page = 0
            dup_in_page = 0

            for book in docs:
                title = book.get("title", "Без названия")
                authors = book.get("author_name", [])
                author_str = ", ".join(authors) if authors else "Unknown"

                # Ключ дедупликации
                key_tuple = (normalize(title), normalize(author_str))
                if key_tuple in seen:
                    dup_in_page += 1
                    continue
                seen.add(key_tuple)
                all_books.append(book)
                new_in_page += 1

            duplicates_total += dup_in_page
            print(f"   Получено {len(docs)} | новых: {new_in_page} | дублей: {dup_in_page} "
                  f"| всего уник.: {len(all_books)}/{total_expected}")

            # Если уже видели столько уникальных, сколько обещает API — выходим
            if len(all_books) + duplicates_total >= total_expected:
                break

            page += 1
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса на странице {page}: {e}")
            time.sleep(2)
            continue

    print(f"\nСбор завершён. Уникальных: {len(all_books)}, дублей пропущено: {duplicates_total}")

    # ---------- Сохранение в CSV ----------
    filename = "creative_writing_books.csv"
    with open(filename, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Название", "Автор", "Год", "Ссылка", "ISBN", "Издатель", "Язык"])

        for book in all_books:
            title = book.get("title", "Без названия")
            authors = ", ".join(book.get("author_name", [])) or "Unknown"
            year = book.get("first_publish_year", "N/A")
            key = book.get("key", "")
            link = f"https://openlibrary.org{key}" if key else "N/A"
            isbn = ", ".join(book.get("isbn", [])[:3])
            publisher = ", ".join(book.get("publisher", [])[:3])
            language = ", ".join(book.get("language", [])[:3])

            writer.writerow([title, authors, year, link, isbn, publisher, language])

    print(f"=== ГОТОВО. '{filename}' сохранён, строк: {len(all_books)} ===")

scrape_creative_writing()