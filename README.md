# Captioning-bot

A Telegram bot with a neural network for generating image descriptions (image captioning).

### To run the project:
* Install `Python 3.9`
* Clone the repository and install dependencies:
  * `git clone https://github.com/PandaMia/Captioning-bot.git`
  * `cd Captioning-bot`
  * `pip install -r requirements.txt`
* Download the weights for the caption generation model and place them in `source/model`:
  * [Model weights](https://drive.google.com/file/d/1XQiRc67_tngFnuIqjLLQYzKW8MGBexCI/view?usp=sharing)
* Register your Telegram bot via `@BotFather`. Assign the generated bot token to the `TOKEN` variable in `bot.py`.
* Run in the console:
  * `python bot.py`

### Bot example:

<img src="/source/data/example.jpg" width="350">
