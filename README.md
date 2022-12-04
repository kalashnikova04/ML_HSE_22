## Домашнее задание № 1.
1. ***Что было сделано и какие результаты?***<br>
    - **EDA**<br>
        - *Удаление дубликатов*<br>
        В обучающей выборке с целевой переменной: 985 дубликатов;<br>
        В обучающей выборке без целевой переменной: 1159 дубликатов;<br>
        Значит, 174 объекта имеют одинаковые значения признаков, но разные значения таргета.<br>
        Удалено 1159 дубликатов - объем обучающей выборки: (5840, 12).
        - *Преобразование числовых колонок*<br>
        С помощью regex в строчных данных были получены числа. Преобразованы колонки mileage, engine, max_power и torque (признак разбит на torque и max_torque). Привеедены к float.
        - *Обработка пропусков*<br>
        Пропуски оказались только в числовых колонках, они были заполнены медианами с помощью SimpleImputer.
    - **Визуализации**<br>
    Была выявлена наибольшая корреляция с целевой переменной (selling_price) у признака max_power: `0.69`.<br>
    Среди признаков замечена положительная корреляция:<br>
        - engine - max_power;
        - torque - max_power;
        - torque - engine.<br>

        Отрицательная корреляция:
        - mileage - max_power;
        - mileage - engine;
        - mileage - seats.<br>

    - **Модели на вещественных признаках**<br>
    Для более понятной интерпретации результатов помимо r2-score использована метрика RMSE.
        - Линейная регрессия без регуляризации и без стандартизации;<br>

            | Метрика       | train              | test |
            | ------------- |:------------------:| -----:|
            | r2            | `0.6015999418524249` | `0.5995245269229015` |
            | RMSE          | `337930.06771979656` | `479796.55871789163` |

        
        - Линейная регрессия без регуляризации и со стандартизацией;<br>
        Метрики те же, но по значениям весов можно интерпретировать информативность признаков в предсказнии таргета: признак max_power оказался самым информативным.

        - Lasso-регрессия (L1-регуляризация);

            | Метрика       | train              | test |
            | ------------- |:------------------:| -----:|
            | r2            | `0.6015999279877879` | `0.5995032784807341` |
            | RMSE          | `337930.0735999133` | `479809.2870806607` |

            С дефолтными значениями параметра alpha L1-регуляризация не занулила веса.<br> 
            Геометрическая интерпретация: квадратичная функция потерь линейной регрессии (круги) пересекается с функцией штрафа весов (ромб/квадрат) не в вершинах ромба.

        - Lasso-регрессия + GridSearchCV;

            | Метрика       | train              | test |
            | ------------- |:------------------:| -----:|
            | r2            | `0.5989539246267079` | `0.5883950523523898` |
            | RMSE          | `-332073.1317354101` | `486417.801431972` |

        - ElasticNet-регрессия (L1+L2 регуляризация) + GridSearchCV.

            | Метрика       | train              | test |
            | ------------- |:------------------:| -----:|
            | r2            | `0.5988129381886447` | `0.5854212619867558` |
            | RMSE          | `-331125.4216837723` | `488171.7906337503` |

        *Итоги*: все метрики, как для трейна, так и для теста, имеют примерно одинаковые оценки производительности для всех моделей. Т.е. улучшить качество применением простых линейных моделей регрессии с вещественными признаками не получилось.

    - **Модели с категориальными признаками**<br>
        - Кодировка категориальных фичей с помощью OneHotEncoding;<br>
        Размер датасета (5840, 18). 
        - На основе получившихся данных обучена модель Ridge-регрессии. Подобран оптимальный параметра `alpha = 7.135175879396985` с помощью GridSearchCV.

            | Метрика       | train              | test |
            | ------------- |:------------------:| -----:|
            | r2            | `0.6142623433370271` | `0.6474540414387997` |
            | RMSE          | `-318448.24856314226` | `449780.4667678565` |

            Результаты Ridge-регрессии с закодированными категориальными фичами оказались чуть лучше, чем линейные модели на вещественных признаках.

    - **Feature Engineering**<br>
    - Была создана полиномиальная модель (добавлены квадратичные признаки). Результаты в пункте 2.
    - Была создана модель линейной регрессии с отлогарифмированными признаками max_power и torque. Признаки были выбраны по их гистограммам (ненормальное распределение).

        | Метрика       | train              | test |
        | ------------- |:------------------:| -----:|
        | r2            | `0.5920431061597426` | `0.5920431061597426` |
        | RMSE          | `341959.1878748341` | `501762.88341788546` |

        Качество стало хуже, чем раньше.

    - **Бизнесовая метрика**<br>
    Создана метрика, которая считает долю предсказаний, которые отличаются от истинных значений не более, чем на 10% в обе стороны.<br>
    Для лучшей модели результат 27%.

    - **Сервис на FastAPI**<br>
    Реализовано 2 функции:
        - Загрузка одного объекта в формате json через \docs.
        - Загрузка файла .csv с признаками нескольких объектов через \docs.


2. ***Лучшее качество***<br>

    Модель с полиномиальными признаками показала лучшее качество как на трейне, так и на тесте.<br> 
    `r2_score_train = 0.8256871561482652` <br> 
    `r2_score_test = 0.8319842960339204`<br>
    

3. ***Что вызвало проблемы?***<br>
    - Добавление пороговых признаков в Feature Engineering смутило, не понятно, как создавать пороги.
    - Неожиданности в таргете
    - FastAPI :)<br>
    Не был сделан Pipeline, я не до разобралась с тем, как его внедрить, из-за чего добавила модель, как в ноутбуке. 