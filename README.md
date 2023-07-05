# AristoсratBot<br>
![ ](https://github.com/GalkaMT/AristokratBot/blob/main/img/waiter.gif)
## Идея работы<br>
Создать сервис, который определяет разные виды бокалов,тарелок,столовых приборов<br>

## Список определяемых классов<br>
| Cutlery | Glass no leg | Glass leg |  Plates |
| --- | --- | --- | --- |
|Dessert fork | Glass for soft drinks | Champagne glass | Bread plate|
|Dessert spoon | Rox | Cognac glass | Deep plate|
|Table fork | Shot | Cordial|  Dining Plate |
|Table knife | Beer glass 2 | Hurricane |
|Tablespoon | Beer glass 4 | Margarita glass |
|Salad fork | Highball | Martini glass |
|Bar spoon | Glass no leg | Wine glass |
|Lobster fork | | Beer glass 1 |  
|Fish fork | | Beer glass 3 |
|Snail fork | | Coupe|
|Fondue fork | | |
|Butter knife | | |
|Snail forceps | | |

## Что под капотом?<br>
[YOLOv8](https://github.com/ultralytics/ultralytics) - семейство моделей обнаружения объектов на базе YOLO от Ultralytics<br>
В данном проекте исспользована модель YOLOv8m, дообученная на 5 датасетах:
1. Mother - нейросеть, различающая метаклассы (тарелки, столовые приборы, бокалы с ножкой и без нее);
2. Plates - нейросеть, отвечающая за распознавание классов метакласса тарелки;
3. Glass no leg - нейросеть, отвечающая за распознавание классов метакласса бокалы без ножки;
4. Glass leg - нейросеть, отвечающая за распознавание классов метакласса бокалы с ножкой;
5. Cutlery - нейросеть, отвечающая за распознавание классов метакласса столовые приборы.
![ ](img/Снимок.png)

## Использование
![ ](img/video_2023-06-08_12-36-47.gif)

## Над проектом работали<br>
[IvaElen](https://github.com/IvaElen) -сбор и разметка датасета, обучение моделей<br>
[GalkaMT](https://github.com/GalkaMT) --сбор и разметка датасета, обучение моделей, оформление репозитория<br>
[AlexeyPratsevityi](https://github.com/AlexeyPratsevityi) - написание телеграм-бота, финальная архитектура<br>
