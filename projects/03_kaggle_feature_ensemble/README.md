# Project 03: Elo Merchant Category Recommendation

Полный ML-пайплайн для завершённого соревнования Kaggle [Elo Merchant Category Recommendation](https://www.kaggle.com/competitions/elo-merchant-category-recommendation).

## Задача и результат

- **Тип задачи:** регрессия.
- **Target:** loyalty score клиента (`target`).
- **Метрика Kaggle и локальной валидации:** RMSE, меньше - лучше.
- **Локальный OOF RMSE финального ансамбля:** **3.64969** на `KFold(n_splits=5, shuffle=True, random_state=42)`.
- **Public leaderboard:** будет добавлен после загрузки `submissions/submission.csv` в Kaggle под аккаунтом автора.
- **Скриншот leaderboard:** поместить в `reports/` после submission; на нём должны быть видны username и score.

## Подход

1. Проведено EDA train/test: размер, уникальность `card_id`, пропуски, распределение и выбросы target.
2. Транзакции из historical и new period агрегированы до уровня `card_id` отдельно, без использования target.
3. Созданы признаки интенсивности покупок, давности операций, рассрочки, разнообразия продавцов, авторизаций, профиля продавца и динамики между двумя периодами.
4. Все модели сравниваются на одинаковых 5 фолдах по RMSE; для бустингов используется early stopping.
5. Подбор LightGBM выполнен Optuna: 20 trial на 3-fold CV, лучший набор подтверждён на 5-fold CV.
6. Финальный прогноз - неотрицательный OOF-blend LightGBM, CatBoost и XGBoost. Веса подбираются только по out-of-fold предсказаниям.

## Итерации

| Вариант | 5-fold RMSE | Вывод |
| --- | ---: | --- |
| DummyRegressor (mean) | 3.84957 | Нижняя граница качества. |
| Ridge + median imputation + scaling | 4.52930 | Нестабилен на одном фолде; линейная модель плохо описывает зависимости. |
| LightGBM, ручная регуляризация | 3.65551 | Сильная базовая бустинговая модель. |
| CatBoost, regularized | 3.65434 | Близкий, но немного иной профиль ошибок. |
| XGBoost, regularized | 3.65282 | Лучшая одиночная модель до Optuna. |
| LightGBM после Optuna | 3.65249 | Небольшое подтверждённое улучшение. |
| **OOF weighted blend** | **3.64969** | Финальный локальный результат. |

## Что сработало

- Агрегация обоих транзакционных периодов до одной строки на карту.
- Разделение исторических и новых операций и признаки их динамики.
- Регуляризованные бустинги, ранняя остановка и сравнение на одинаковых фолдах.
- OOF-блендинг: он улучшил лучшую одиночную модель, хотя ошибки бустингов сильно коррелируют.

## Что не сработало или не вошло в финал

- Ridge показал сильную нестабильность, поэтому в ансамбль не включён.
- ExtraTrees был остановлен: 500 деревьев на пяти фолдах потребовали несоразмерно много CPU, а ожидаемый вклад в ансамбль был ниже, чем у трёх бустингов.
- Не используются target encoding и `card_id` как числовой признак: это снизило бы надёжность решения и создало бы риск утечки.

## Структура

```text
03_kaggle_feature_ensemble/
  data/                       # исходные файлы Kaggle, не попадают в Git
  docs/homework2.pdf          # условие задания
  notebooks/solution.ipynb    # EDA, обучение, подбор и submission
  reports/                    # скриншот Kaggle leaderboard после загрузки
  submissions/submission.csv  # готовый файл для Kaggle
  src/elo_features.py         # воспроизводимый feature engineering
  src/elo_training.py         # CV, OOF-предсказания и блендинг
  README.md
  requirements.txt
```

## Воспроизведение

1. Скачайте данные соревнования и поместите CSV-файлы в `data/`.
2. Установите зависимости:

```bash
cd projects/03_kaggle_feature_ensemble
pip install -r requirements.txt
```

3. Откройте и выполните сверху вниз [notebooks/solution.ipynb](notebooks/solution.ipynb). Первый запуск создаёт признаки в `artifacts/`, повторные используют кэш.
4. Загрузите `submissions/submission.csv` на Kaggle и добавьте public score, место и screenshot в этот README/`reports/`.

`data/` и `artifacts/` намеренно исключены из Git: исходные файлы велики, а признаки можно воспроизвести из кода.
