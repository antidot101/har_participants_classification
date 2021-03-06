# Классификация участников по данным акселерометра

Даны два участника, которые на протяжении нескольких дней, даже в момент сна, носили браслеты, записывающие показания акселерометра. Id участников 01-0 и 02-0. Файлы 01-0 и 02-0 содержат данные этих участников. Необходимо построить классификатор, с помощью которого возможно было бы классифицировать участников между собой на основе данных акселерометра.   
Stages:  
1.	EDA in Jupyter Notebook.
2.	Model selection.
3.	Model fitting, validation, estimation.
4.	Model comparing and reasonable choosing.  

Исходные данные (с частотой 100 Гц) ресемплированы до частоты дискретизации 10 Гц.  
Рассматриваются следующие классические модели:  
- линейные (логистическая регрессия, метод опорных векторов);
- ансамблевые (случайный лес, градиентный бустинг),  

а также нейронная сеть.
Интерес представляет свёрточная нейронная сеть для подобных задач классификации (CNN пока не представлена).  
Для настройки гиперпараметров классификаторов используется сеточный поиск с k-блочной кросс-валидацией и оценкой по accuracy.  
Для линейных классификаторов проведена нормализация значений признаков методом стандартизации.  
Тренировочные и тестовые данные разделены в соотношении 70/30.  
