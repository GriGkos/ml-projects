# ML Projects

Учебное портфолио по машинному обучению: от работы с `NumPy` и `pandas` до полных решений соревнований Kaggle. Каждый проект оформлен как самостоятельное исследование: с постановкой задачи, проверкой качества, выводами и инструкцией по запуску.

## Проекты

| № | Проект | Задача и основные идеи | Результат |
| --- | --- | --- | --- |
| 00 | [NumPy и pandas](https://github.com/GriGkos/ml-projects/tree/project/00-numpy-pandas/projects/00_numpy_pandas) | Первичная работа с таблицами: загрузка Excel, проверка структуры, фильтрация и группировки. | Освоены базовые операции с `DataFrame` и `ndarray`. |
| 01 | [ML Bricks](https://github.com/GriGkos/ml-projects/tree/project/01-ml-bricks/projects/01_ml_bricks) | Базовые элементы ML-пайплайна: train/test split, preprocessing, метрики, CV и подбор параметров. | Все задания и автопроверки выполнены. |
| 02 | [Online Shoppers Purchasing Intention](https://github.com/GriGkos/ml-projects/tree/project/02-ml-basics/projects/02_ml_basics) | Предсказание покупки по поведению пользователя в интернет-магазине. EDA, гипотезы, feature selection и сравнение моделей. | Лучший ROC-AUC на тесте: **0.937** у Gradient Boosting. |
| 03 | [Elo Merchant Category Recommendation](https://github.com/GriGkos/ml-projects/tree/project/03-kaggle-feature-ensemble/projects/03_kaggle_feature_ensemble) | Регрессия лояльности клиента по истории транзакций. Чтение крупных таблиц чанками, feature engineering, fold-safe target encoding и ансамбли. | Private RMSE: **3.60826**, 82-е место из 4 111, топ **1.99%**. |
| 04 | [Home Credit Default Risk](https://github.com/GriGkos/ml-projects/tree/project/04-home-credit-scoring/projects/04_home_credit_scoring) | Кредитный скоринг по заявке и истории клиента. Полный pipeline, OOF-валидация, CatBoost, LightGBM, MLP и cross-fitted blend. | Private ROC-AUC: **0.79510**, OOF ROC-AUC: **0.79405**. |

## Что внутри

### 00. NumPy и pandas

Небольшой вводный проект, в котором отработаны типичные действия аналитика: загрузка Excel-таблицы, просмотр структуры данных, удаление служебного поля, группировки и фильтрация записей. Это основа для следующих исследований.

### 01. ML Bricks

Практика ключевых компонентов модели классификации на `load_breast_cancer`: ручной стратифицированный split, импутация без утечки, One-Hot Encoding, расчёт метрик, cross-validation и `GridSearchCV`. Отдельно разобраны baseline через `DummyClassifier` и важность признаков дерева решений.

### 02. Online Shoppers Purchasing Intention

Первое полноценное исследование классификации. Для сессий интернет-магазина проверены пропуски, дубликаты, распределения, взаимосвязи признаков и дисбаланс классов. Гипотезы формулируются после соответствующих наблюдений в EDA и проверяются на одних и тех же CV-фолдах.

Сравниваются Logistic Regression, Random Forest и Gradient Boosting. Все преобразования выполнены внутри `Pipeline`, поэтому scaling, кодирование и отбор признаков не используют информацию из валидационных фолдов.

### 03. Elo Merchant Category Recommendation

Полное решение регрессионного соревнования Kaggle с таблицами транзакций объёмом в десятки миллионов строк. Признаки строятся из historical и new периодов, агрегации рассчитываются чанками до уровня карты. Сравниваются LightGBM, XGBoost, CatBoost, target encoding и два способа ансамблирования.

Финальная модель выбрана не только по среднему OOF RMSE: преимущество сложного ансамбля над XGBoost seed bagging было проверено парным bootstrap. Так как доверительный интервал разницы включил ноль, для submission оставлена более простая и устойчивая модель.

### 04. Home Credit Default Risk

Наиболее крупное исследование в репозитории. Модель предсказывает риск проблем с погашением кредита и последовательно показывает вклад каждого источника информации: текущей заявки, исторических таблиц и модели.

В проекте есть EDA, агрегирование шести исторических таблиц, feature engineering, сравнение baseline, LightGBM, CatBoost и MLP, анализ калибровки, permutation importance, SHAP и ошибок модели. Итоговый CatBoost подтверждён как OOF-валидацией, так и private leaderboard. В ветке также лежат PDF-отчёт, графики и готовые CSV для Kaggle.

## Подход к работе

Во всех исследовательских проектах я придерживаюсь одних принципов:

- отделяю EDA, построение признаков, обучение и интерпретацию результата;
- использую `Pipeline` или fold-safe преобразования, чтобы исключить утечку данных;
- сравниваю модели на одинаковых фолдах кросс-валидации;
- не выбираю финальную модель только по одной метрике: проверяю разницу между экспериментами, стабильность по фолдам и внешний результат на Kaggle;
- сохраняю итоговые выводы, воспроизводимые ноутбуки, README и submission рядом с решением.

## Как открыть проект

Каждая работа хранится в отдельной ветке. Чтобы посмотреть конкретный проект локально:

```bash
git clone https://github.com/GriGkos/ml-projects.git
cd ml-projects
git switch project/04-home-credit-scoring
```

Затем откройте README в папке `projects/<название-проекта>/`: в нём перечислены данные, зависимости, структура файлов и инструкция по воспроизведению. Исходные данные Kaggle намеренно не добавляются в Git, но для каждого проекта указано, куда их положить.

## Стек

`Python` · `pandas` · `NumPy` · `scikit-learn` · `LightGBM` · `XGBoost` · `CatBoost` · `Optuna` · `PyTorch` · `SHAP` · `Jupyter`

## Структура репозитория

```text
ml-projects/
|-- README.md                         # эта страница
|-- projects/
|   |-- 00_numpy_pandas/              # ветка project/00-numpy-pandas
|   |-- 01_ml_bricks/                 # ветка project/01-ml-bricks
|   |-- 02_ml_basics/                 # ветка project/02-ml-basics
|   |-- 03_kaggle_feature_ensemble/   # ветка project/03-kaggle-feature-ensemble
|   `-- 04_home_credit_scoring/       # ветка project/04-home-credit-scoring
|-- requirements.txt
`-- .gitignore
```

Папки проектов показаны здесь как карта репозитория: фактическое содержимое каждой из них находится в соответствующей ветке.
