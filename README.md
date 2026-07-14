# ML Projects Repository

Репозиторий для учебных проектов по курсу ML.

## Как устроен репозиторий

- Все проекты хранятся в одном репозитории.
- Каждый проект оформляется в отдельной ветке.
- Для каждой темы создаётся отдельная папка внутри `projects/`.
- Нумерация начинается с `00`, если первый проект был вводным.

## Структура

```text
projects/
  00_numpy_pandas/
    00_numpy_pandas.ipynb
    README.md
    data/
      credit_data.xlsx
  01_ml_bricks/
    01_ml_bricks.ipynb
    README.md
```

## Текущие проекты

- `00_numpy_pandas` - базовая практика по `NumPy` и `pandas`
- `01_ml_bricks` - тренировка базовых кирпичиков ML-пайплайна

## Рекомендуемый workflow

1. Создать ветку под новый проект:
   `git checkout -b project/<project-name>`
2. Выполнить задание и оформить решение в папке `projects/<project-name>`.
3. Закоммитить изменения:
   `git add .`
   `git commit -m "Add project <project-name>"`
4. Запушить ветку и приложить ссылку на неё в форме сдачи.

## Зависимости

Для запуска ноутбуков достаточно установить:

```text
numpy
pandas
scikit-learn
openpyxl
jupyter
```
