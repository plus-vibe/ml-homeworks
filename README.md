# DS Infrastructure Homework

Учебный Python-проект для домашнего задания №1 «Инфраструктура проекта».

## Технологии

- Python 3.12
- uv — управление зависимостями
- Ruff — линтер и форматтер
- mypy — статическая проверка типов
- pre-commit — автоматические проверки перед коммитом
- GitHub Actions — автоматическая проверка pull request

## Установка и запуск

### Требования

- Linux, macOS или Windows с WSL2
- Git
- Make
- uv

### Клонирование репозитория

```bash
git clone git@github.com:plus-vibe/ml-homeworks.git
cd ml-homeworks
```

### Установка зависимостей

```bash
make setup
```

Команда создаёт виртуальное окружение, устанавливает зависимости из `uv.lock` и активирует Git-хуки pre-commit.

### Запуск приложения

```bash
uv run ds-infrastructure-hw
```

## Проверка качества кода

Запустить все проверки:

```bash
make check
```

Отдельные команды:

| Команда | Назначение |
|---|---|
| `make lint` | Проверка Ruff |
| `make format-check` | Проверка форматирования |
| `make typecheck` | Статическая проверка типов |
| `make format` | Автоматическое форматирование и исправление поддерживаемых ошибок |
| `make setup` | Подготовка окружения |
| `make check` | Все проверки без изменения файлов |

## Git workflow

Разработка ведётся в отдельных ветках. Изменения в `main` вносятся через pull request.

Перед коммитом pre-commit автоматически проверяет стиль и типы. Коммит с ошибками блокируется.

## Структура проекта

```text
.
├── .github/
│   └── workflows/
│       └── checks.yml
├── .pre-commit-config.yaml
├── .python-version
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
└── src/
    └── ds_infrastructure_hw/
        ├── __init__.py
        └── calculator.py
```
