import requests, csv, time, re

def normalize(s):
    if not s: return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s

# Список немецких поисковых запросов
GERMAN_QUERIES = [
    "kreatives schreiben",
    "kreativ schreiben",
    "Schreibwerkstatt",
    "Schreiben lernen",
    "Creative Writing",   # нем. издания часто сохраняют англ. термин
]

# Слова-индикаторы немецкого языка в названии/описании
GERMAN_HINTS = [
    "schreiben", "schreib", "werkstatt", "kursbuch",
    "anleitung", "ratgeber", "lehrbuch", "übung",
    "autor", "autoren", "literarisches", "erzählen",
]

def scrape_kreatives_schreiben_deutsch():
    print("=== ЗАПУСК СКРЕПЕРА (немецкоязычные книги по креативному письму) ===\n")

    url = "https://openlibrary.org/search.json"
    all_books = []
    seen = set()
    duplicates_total = 0
    rejected_language = 0
    rejected_no_german_hint = 0

    for query in GERMAN_QUERIES:
        print(f"\n--- Поиск по запросу: '{query}' ---")
        page = 1
        max_pages = 30

        while page <= max_pages:
            params = {
                "q": query,
                "limit": 100,
                "page": page,
            }
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if page == 1:
                    print(f"   numFound = {data.get('numFound', 0)}")

                docs = data.get("docs", [])
                if not docs:
                    print(f"   стр. {page} пуста.")
                    break

                for book in docs:
                    # === ФИЛЬТР 1: язык ===
                    langs = book.get("language", [])
                    # Open Library хранит языки как коды: ['ger', 'eng']
                    # Оставляем только книги с 'ger' в списке
                    if "ger" not in langs:
                        rejected_language += 1
                        continue

                    # === ФИЛЬТР 2: немецкие маркеры в названии ===
                    title = book.get("title", "")
                    subtitle = book.get("subtitle", "")
                    title_full = f"{title} {subtitle}".lower()

                    has_german_hint = any(h in title_full for h in GERMAN_HINTS)
                    # Если нет маркера — пропускаем (защита от англ. книг с кодом 'ger'
                    # из-за ошибочной разметки, такое бывает)
                    if not has_german_hint:
                        rejected_no_german_hint += 1
                        continue

                    # === Дедупликация ===
                    authors = book.get("author_name", [])
                    author_str = ", ".join(authors) if authors else "Unknown"
                    key_tuple = (normalize(title), normalize(author_str))
                    if key_tuple in seen:
                        duplicates_total += 1
                        continue
                    seen.add(key_tuple)
                    all_books.append(book)

                print(f"   стр. {page}: получено {len(docs)}, "
                      f"прошло все фильтры: {len([b for b in docs if 'ger' in b.get('language', [])])}")

                page += 1
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                print(f"   ошибка стр. {page}: {e}")
                time.sleep(2)
                continue

    print(f"\n=== ИТОГИ ===")
    print(f"   Прошло все фильтры (немецкий язык + маркер): {len(all_books)}")
    print(f"   Отклонено по языку (нет 'ger'):                {rejected_language}")
    print(f"   Отклонено по отсутствию немецких маркеров:     {rejected_no_german_hint}")
    print(f"   Дублей пропущено:                              {duplicates_total}")

    # ---------- Сохранение ----------
    filename = "kreatives_schreiben_deutsch.csv"
    with open(filename, mode="w", newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Название", "Подзаголовок", "Автор", "Год",
                         "Языки", "Ссылка", "ISBN", "Издатель"])

        for book in all_books:
            title = book.get("title", "Без названия")
            subtitle = book.get("subtitle", "")
            authors = ", ".join(book.get("author_name", [])) or "Unknown"
            year = book.get("first_publish_year", "N/A")
            langs = ", ".join(book.get("language", []))
            key = book.get("key", "")
            link = f"https://openlibrary.org{key}" if key else "N/A"
            isbn = ", ".join(book.get("isbn", [])[:3])
            publisher = ", ".join(book.get("publisher", [])[:3])

            writer.writerow([title, subtitle, authors, year,
                             langs, link, isbn, publisher])

    print(f"\n=== ГОТОВО. '{filename}' сохранён, строк: {len(all_books)} ===")

scrape_kreatives_schreiben_deutsch()