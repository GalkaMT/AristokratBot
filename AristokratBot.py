import asyncio

from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.dispatcher.handler import CancelHandler
from aiogram.dispatcher.middlewares import BaseMiddleware
from aiogram.types import InputMedia
from aiogram.types import InputFile
import config
import predict_parall
from predict_parall import predict_img
from ultralytics import YOLO
import os
HOME = os.getcwd()
print(HOME)
model_mother = YOLO('/home/alexeyprats/Telegrambots/final_poject_bot/best_mother_m.pt')
model_cul = YOLO('/home/alexeyprats/Telegrambots/final_poject_bot/best_medium_cul.pt')
model_leg = YOLO('/home/alexeyprats/Telegrambots/final_poject_bot/best_medium_leg.pt')
model_NO_leg = YOLO('/home/alexeyprats/Telegrambots/final_poject_bot/best_no_leg.pt')
model_plate = YOLO('/home/alexeyprats/Telegrambots/final_poject_bot/best_medium_plate.pt')
path_img = "/home/alexeyprats/Telegrambots/final_poject_bot/venv/photos/photo_2023-06-06_11-47-46.jpg"




bot = Bot(token='')
dp = Dispatcher(bot, storage=MemoryStorage())




@dp.message_handler(commands=['start'],state=None)
async def start(message: types.Message):
    await message.answer('Привет! Пришли мне фотографии приборов, которых ты не знаешь')



class AlbumMiddleware(BaseMiddleware):
    """This middleware is for capturing media groups."""
    album_data: dict = {}

    def __init__(self, latency: int | float = 0.01):
        """
        You can provide custom latency to make sure
        albums are handled properly in highload.
        """
        self.latency = latency
        super().__init__()

    async def on_process_message(self, message: types.Message, data: dict):
        if not message.media_group_id:
            return

        try:
            self.album_data[message.media_group_id].append(message)
            raise CancelHandler()  # Tell aiogram to cancel handler for this group element
        except KeyError:
            self.album_data[message.media_group_id] = [message]
            await asyncio.sleep(self.latency)

            message.conf["is_last"] = True
            data["album"] = self.album_data[message.media_group_id]

    async def on_post_process_message(self, message: types.Message, result: dict, data: dict):
        """Clean up after handling our album."""
        if message.media_group_id and message.conf.get("is_last"):
            del self.album_data[message.media_group_id]


@dp.message_handler(content_types=['photo'], state=None)
async def get_message(message: types.Message, album: list[types.Message] = None):
    print(message)
    if not album:
        album = [message]

    media_group = types.MediaGroup()
    for ind, obj in enumerate(album):
        if obj.photo:
            file_id = obj.photo[-1].file_id
            model_mother, model_cul, model_leg, model_NO_leg, model_plate
            await obj.photo[-1].download(f'/home/alexeyprats/Telegrambots/final_poject_bot/photos/photo{ind}.jpg')
            file_predict = predict_img(f'/home/alexeyprats/Telegrambots/final_poject_bot/photos/photo{ind}.jpg',
                                       model_mother, model_cul, model_leg, model_NO_leg, model_plate)
            photo = InputFile(file_predict)
        else:
            file_id = obj[obj.content_type].file_id
        try:
            if obj == album[-1]:
                media_group.attach(InputMedia(media=photo,
                                              type=obj.content_type))
            else:
                media_group.attach(InputMedia(media=photo, type=obj.content_type))
        except ValueError:
            return await message.answer("This type of album is not supported by aiogram.")
    await bot.send_media_group(message.chat.id, media_group)





if __name__ == '__main__':
    dp.middleware.setup(AlbumMiddleware())
    executor.start_polling(dp, skip_updates=True)
