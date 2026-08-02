# Project 03: Kaggle Feature Engineering & Ensemble

Практический проект по классическому машинному обучению на завершенном соревновании Kaggle. Цель - построить воспроизводимый пайплайн, улучшить baseline с помощью feature engineering и ансамбля и зафиксировать результат на публичном лидерборде.

## Статус

Соревнование пока не выбрано. После выбора нужно заполнить этот раздел до начала моделирования:

- название и ссылка на Kaggle;
- тип задачи и целевая переменная;
- метрика соревнования;
- публичный score и место на лидерборде;
- дата последнего submission.

Подходящие варианты из условия: `Spaceship Titanic`, `Porto Seguro Safe Driver Prediction`, `Santander Customer Transaction Prediction`, `Home Credit Default Risk`, `Elo Merchant Category Recommendation`, `Santander Value Prediction`.

## Что должно быть в решении

- EDA: пропуски, распределения, аномалии, корреляции и выводы.
- Минимум 3-5 осмысленных новых признаков с объяснением, зачем они нужны.
- Не менее 3-4 разных моделей и подбор гиперпараметров.
- Кросс-валидация по метрике соревнования.
- Ансамбль лучших моделей: weighted voting, blending или stacking.
- Финальный `submission.csv` в папке `submissions/`.
- Скриншот лидерборда с username и score в `reports/`.
- Описание удачных и неудачных итераций.

## Структура

```text
03_kaggle_feature_ensemble/
  data/                 # исходные данные Kaggle, не попадают в Git
  docs/homework2.pdf    # условие задания
  notebooks/
    solution.ipynb      # основной notebook
  reports/              # скриншот leaderboard
  submissions/          # финальный submission.csv
  README.md
  requirements.txt
```

## Журнал итераций

| Версия | Изменение | CV score | Public LB | Вывод |
| --- | --- | ---: | ---: | --- |
| 0 | Baseline | - | - | Заполнить после выбора соревнования. |

## Запуск

```bash
cd projects/03_kaggle_feature_ensemble
pip install -r requirements.txt
jupyter notebook notebooks/solution.ipynb
```

Датасет нужно скачать со страницы выбранного соревнования Kaggle и поместить в `data/`. Перед финальной сдачей в README должны быть актуальные score, место в лидерборде, краткий путь от baseline до финального ансамбля и ссылка на соревнование.
