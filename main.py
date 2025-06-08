import asyncio
import os
from io import BytesIO
import speech_recognition as sr
from pydub import AudioSegment
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, BufferedInputFile
from neural import ChatNeural
from gtts import gTTS

bot = Bot(token="<YOU_BOT_TOKEN>")
dp = Dispatcher()
neural = ChatNeural()


def async_to_sync(f):
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(f, *args, **kwargs)
    return wrapper


async def convert_voice_to_text(bot: Bot, voice: types.Voice) -> str:
    recognizer = sr.Recognizer()
    file = await bot.get_file(voice.file_id)
    ogg_data = BytesIO()
    await bot.download_file(file.file_path, destination=ogg_data)
    ogg_data.seek(0)

    try:
        # Конвертируем OGG в WAV с помощью pydub
        audio = AudioSegment.from_ogg(ogg_data)
        wav_data = BytesIO()
        audio.export(wav_data, format="wav")
        wav_data.seek(0)

        with sr.AudioFile(wav_data) as source:
            print("Аудиофайл успешно загружен и конвертирован.")
            audio_data = recognizer.record(source)
            print("Аудио записано для распознавания.")
            return recognizer.recognize_google(audio_data, language="ru-RU")
    except sr.UnknownValueError:
        print("Google Speech Recognition не смог распознать аудио.")
    except sr.RequestError as e:
        print(f"Ошибка запроса к Google Speech Recognition: {e}")
    except Exception as e:
        print(f"Ошибка при распознавании голоса: {e}")
    return None

async def text_to_voice(text: str) -> BytesIO:
    tts = gTTS(text=text, lang='ru')
    voice_data = BytesIO()
    tts.write_to_fp(voice_data)
    voice_data.seek(0)
    return voice_data


@dp.message(Command("start"))
async def start_command(message: Message):
    await message.answer("Привет! Отправьте голосовое или текстовое сообщение.")


@dp.message(lambda message: message.voice is not None)
async def handle_voice_message(message: Message):
    recognized_text = await convert_voice_to_text(bot, message.voice)

    if not recognized_text:
        await message.answer("⚠️ Не удалось обработать голосовое сообщение. Проверьте, установлен ли ffmpeg.")
        return

    response = await get_neural_response(recognized_text)
    voice_data = await text_to_voice(response)
    await message.answer_voice(voice=BufferedInputFile(voice_data.getvalue(), filename="response.mp3"))


@dp.message()
async def handle_text_message(message: Message):
    response = await get_neural_response(message.text)
    await message.answer(response)


@async_to_sync
def get_neural_response(text: str) -> str:
    return neural.get_response(text)


async def main():
    print("Выберите режим работы:\n1 - Телеграм бот\n2 - Консольное тестирование")
    mode = input("Введите номер режима: ")

    if mode == "1":
        await dp.start_polling(bot)
    elif mode == "2":
        print("Консольный режим...")
        while True:
            text = input("Вы: ")
            if text.lower() in ('выход', 'exit', 'quit'):
                break
            response = get_neural_response(text)
            print("Бот:", response)
    else:
        print("Неверный режим")

if __name__ == "__main__":
    asyncio.run(main())
