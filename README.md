# ML Projects Repository

Репозиторий для учебных проектов по курсу ML.

## Как устроен репозиторий

- Все проекты хранятся в одном репозитории.
- Каждая работа оформляется в отдельной ветке.
- Для каждой темы создаётся отдельная папка внутри `projects/`.

## Структура

```text
projects/
  01_numpy_pandas/
    01_numpy_pandas.ipynb
    README.md
    data/
      credit_data.xlsx
```

## Текущие проекты

- `01_numpy_pandas` - базовая практика по `NumPy` и `pandas`

## Рекомендуемый workflow

1. Создать ветку под новый проект:
   `git checkout -b project/01-numpy-pandas`
2. Выполнить задание и оформить решение в папке `projects/<topic_name>`.
3. Закоммитить изменения:
   `git add .`
   `git commit -m "Add project 01: numpy and pandas"`
4. Запушить ветку и приложить ссылку на неё в форме сдачи.

## Зависимости

Для запуска ноутбуков достаточно установить:

```text
numpy
pandas
openpyxl
jupyter
```
