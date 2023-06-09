# AristoсratBot<br>
![ ](https://github.com/GalkaMT/AristokratBot/blob/main/img/waiter.gif)
## Идея работы<br>
Создать сервис, который определяет разные виды бокалов,тарелок,столовых приборов<br>




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
[IvaElen](https://github.com/IvaElen)<br>
[GalkaMT](https://github.com/GalkaMT)<br>
[AlexeyPratsevityi](https://github.com/AlexeyPratsevityi)<br>



