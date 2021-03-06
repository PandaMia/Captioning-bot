# Captioning-bot

Telegram-бот с нейронной сетью для генерации описаний к изображениям (image captioning).  

### Для запуска проекта необходимо:
* Установить ```Python 3.9```
* Клонировать репозиторий и установить зависимости:
  * ```git clone https://github.com/PandaMia/Captioning-bot.git```
  * ```cd Captioning-bot```
  * ```pip install -r requirements.txt```
* Скачать веса для модели генерации описаний и разместить по пути ```source/model```:  
  * [Веса модели](https://drive.google.com/file/d/1XQiRc67_tngFnuIqjLLQYzKW8MGBexCI/view?usp=sharing)
* Зарегистрировать своего телеграм-бота через @BotFather. Полученный токен бота присвоить в переменную ```TOKEN``` в файле ```bot.py```.
* В консоли вызвать:  
  * ```python bot.py```

### Пример работы бота:  

<img src="/source/data/example.jpg" width="350">
