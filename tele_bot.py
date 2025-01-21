from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_API = os.getenv("TELEGRAM_API_KEY")

import requests

def check_internet():
    try:
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("Internet connection is available.")
            return True
        else:
            print(f"Internet check failed with status code: {response.status_code}")
            return False
    except requests.ConnectionError as e:
        print(f"No internet connection: {e}")
        return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hello {update.effective_user.first_name}')
    
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    response = await query_backend(user_message)
    await update.message.reply_text(response)

async def query_backend(user_message):
    async with aiohttp.ClientSession() as session:
        async with session.post("https://carboncredits-tccv-edu-chatbot.hf.space/query", json={"user_message": user_message}) as resp:
            response_data = await resp.json()
            return response_data["response"]

def main() -> None:
    check_internet()
    app = ApplicationBuilder().token(TELEGRAM_API).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()